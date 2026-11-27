// Optional fused enumeration / guarded Fisher / reduction for process workers.
// No NumPy C ABI, global mutable cache, threads, or replacement LAPACK solver.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <list>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
struct PythonError {};
struct Unsupported {};
struct ReleasedGIL {
    PyThreadState* state;
    ReleasedGIL() : state(PyEval_SaveThread()) {}
    ~ReleasedGIL() { acquire(); }
    void acquire() {
        if (state) {
            PyEval_RestoreThread(state);
            state = nullptr;
        }
    }
    void release() {
        if (!state)
            state = PyEval_SaveThread();
    }
};
struct AcquiredGIL {
    ReleasedGIL& owner;
    explicit AcquiredGIL(ReleasedGIL& unlocked) : owner(unlocked) { owner.acquire(); }
    ~AcquiredGIL() { owner.release(); }
};
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value = nullptr) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    Ref(Ref&& other) noexcept : p(other.p) { other.p = nullptr; }
    Ref& operator=(Ref&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(p);
            p = other.p;
            other.p = nullptr;
        }
        return *this;
    }
    PyObject* release() {
        auto result = p;
        p = nullptr;
        return result;
    }
};
PyObject* checked(PyObject* p) {
    if (!p)
        throw PythonError();
    return p;
}
Py_ssize_t integer(PyObject* p) {
    if (!PyIndex_Check(p))
        throw Unsupported();
    Ref indexed(checked(PyNumber_Index(p)));
    Py_ssize_t value = PyLong_AsSsize_t(indexed.p);
    if (PyErr_Occurred())
        throw PythonError();
    return value;
}
std::vector<Py_ssize_t> integers(PyObject* obj) {
    Ref sequence(checked(PySequence_Fast(obj, "expected an integer sequence")));
    std::vector<Py_ssize_t> result;
    auto size = PySequence_Fast_GET_SIZE(sequence.p);
    result.reserve(size);
    for (Py_ssize_t i = 0; i < size; ++i)
        result.push_back(integer(PySequence_Fast_GET_ITEM(sequence.p, i)));
    return result;
}
struct Matrix {
    Py_buffer buffer{};
    explicit Matrix(PyObject* obj) {
        if (PyObject_GetBuffer(obj, &buffer, PyBUF_STRIDES | PyBUF_FORMAT) < 0)
            throw PythonError();
    }
    ~Matrix() { PyBuffer_Release(&buffer); }
    bool supported() const {
        return buffer.ndim == 2 && buffer.shape[0] == buffer.shape[1] &&
               buffer.itemsize == sizeof(double) && buffer.format &&
               std::strcmp(buffer.format, "d") == 0 && !buffer.suboffsets;
    }
    bool contains(Py_ssize_t index) const { return index >= 0 && index < buffer.shape[0]; }
    double at(Py_ssize_t x, Py_ssize_t y) const {
        double result;
        std::memcpy(&result,
                    static_cast<char*>(buffer.buf) + x * buffer.strides[0] + y * buffer.strides[1],
                    sizeof(result));
        return result;
    }
};
constexpr double eps = std::numeric_limits<double>::epsilon();
bool symmetric(double a, double b) {
    return std::abs(a - b) <= 8 * eps * std::max({1., std::abs(a), std::abs(b)});
}
bool low_order(const Matrix& corr, Py_ssize_t samples, Py_ssize_t x, Py_ssize_t y,
               const std::vector<Py_ssize_t>& condition, double alpha, double& value) {
    if (condition.size() > 1 || samples - static_cast<Py_ssize_t>(condition.size()) - 3 <= 0)
        return false;
    double a = corr.at(x, x), b = corr.at(x, y), d = corr.at(y, y), rb = corr.at(y, x);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(d) || !std::isfinite(rb) ||
        std::min(a, d) <= 0 || !symmetric(b, rb))
        return false;
    double numerator, denominator, determinant, trace_power;
    if (condition.empty()) {
        double trace = a + d;
        trace_power = trace * trace;
        determinant = a * d - b * b;
        numerator = b;
        denominator = a * d;
    } else {
        Py_ssize_t z = condition[0];
        double c = corr.at(x, z), e = corr.at(y, z), f = corr.at(z, z);
        double rc = corr.at(z, x), re = corr.at(z, y);
        if (!std::isfinite(c) || !std::isfinite(e) || !std::isfinite(f) || !std::isfinite(rc) ||
            !std::isfinite(re) || f <= 0 || !symmetric(c, rc) || !symmetric(e, re))
            return false;
        double first = a * f - c * c, second = d * f - e * e;
        if (std::min({first, second, a * d - b * b}) <= 0)
            return false;
        numerator = b * f - c * e;
        denominator = first * second;
        determinant = a * second - b * numerator + c * (b * e - c * d);
        double trace = a + d + f;
        trace_power = trace * trace * trace;
    }
    if (!std::isfinite(trace_power) || !std::isfinite(denominator) || denominator <= 0 ||
        determinant <= trace_power * 1e-6)
        return false;
    double r = numerator / std::sqrt(denominator);
    r = std::min(.9999999, std::max(-.9999999, r));
    double z = .5 * std::log((1 + r) / (1 - r));
    double dof_root = std::sqrt(static_cast<double>(samples - condition.size() - 3));
    value = std::erfc(dof_root * std::abs(z) / std::sqrt(2.));
    // Same conservative legacy-boundary guard as independence_tests.py.
    double uncertainty = (256 * eps * (trace_power / determinant) * std::max(1., dof_root) /
                          std::max(1e-15, 1 - r * r));
    return !(std::abs(value - alpha) <= uncertainty);
}
struct Key {
    Py_ssize_t a, b;
    std::vector<Py_ssize_t> condition;
    bool operator==(const Key& other) const {
        return a == other.a && b == other.b && condition == other.condition;
    }
};
struct Hash {
    size_t operator()(const Key& key) const {
        size_t hash = static_cast<size_t>(key.a) * 1000003 + key.b;
        for (auto v : key.condition)
            hash = (hash ^ static_cast<size_t>(v)) * 1000003;
        return hash;
    }
};
Ref condition_object(const std::vector<Py_ssize_t>& values) {
    Ref result(checked(PyTuple_New(values.size())));
    for (size_t i = 0; i < values.size(); ++i)
        PyTuple_SET_ITEM(result.p, i, checked(PyLong_FromSsize_t(values[i])));
    return result;
}
Ref key_object(const Key& key) {
    auto condition = condition_object(key.condition);
    return Ref(checked(Py_BuildValue("nnN", key.a, key.b, condition.release())));
}
struct Event {
    Key key;
    double value = 0;
    bool ready = false;
    bool promotion = false;
};
using EventPtr = std::shared_ptr<Event>;
void append(std::deque<EventPtr>& events, const EventPtr& event, Py_ssize_t retain) {
    if (retain <= 0)
        return;
    events.push_back(event);
    if (events.size() > static_cast<size_t>(retain))
        events.pop_front();
}
Ref events_object(const std::deque<EventPtr>& events, bool promotions_only = false) {
    Ref result(checked(PyList_New(0)));
    for (const auto& event : events) {
        if (!event->ready || (promotions_only && !event->promotion))
            continue;
        auto key = key_object(event->key);
        Ref row(checked(Py_BuildValue("Nd", key.release(), event->value)));
        if (PyList_Append(result.p, row.p) < 0)
            throw PythonError();
    }
    return result;
}
struct Query {
    size_t edge;
    int direction;
    bool common;
    Key key;
    EventPtr event;
};
struct State {
    double pmax[2] = {-std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()};
    std::vector<std::pair<int, std::vector<Py_ssize_t>>> separators;
    std::set<Py_ssize_t> members;
};
void consume(State& state, int direction, const std::vector<Py_ssize_t>& condition, bool common,
             double value, double alpha, bool fci) {
    if (value > state.pmax[direction])
        state.pmax[direction] = value;
    if (common && value > state.pmax[1])
        state.pmax[1] = value;
    if (fci ? value >= alpha : value > alpha) {
        state.separators.emplace_back(direction, condition);
        state.members.insert(condition.begin(), condition.end());
    }
}
void put(PyObject* dict, const char* name, PyObject* value) {
    Ref owned(checked(value));
    if (PyDict_SetItemString(dict, name, owned.p) < 0)
        throw PythonError();
}
void count(PyObject* dict, const char* name, size_t value) {
    put(dict, name, PyLong_FromSize_t(value));
}

PyObject* level_impl(PyObject* args) {
    PyObject *pairs_obj, *reference_obj, *corr_obj, *seed_obj, *callback;
    Py_ssize_t samples, depth, retain, window;
    double alpha;
    int cache_enabled, outer_enabled, fci, native_math;
    if (!PyArg_ParseTuple(args, "OOOnndOppnpOnp", &pairs_obj, &reference_obj, &corr_obj, &samples,
                          &depth, &alpha, &seed_obj, &cache_enabled, &outer_enabled, &retain, &fci,
                          &callback, &window, &native_math))
        return nullptr;
    Matrix corr(corr_obj);
    if (!corr.supported() || depth < 0 || retain < 0 || window <= 0 || !PyDict_Check(seed_obj))
        throw Unsupported();
    Ref refs(checked(PySequence_Fast(reference_obj, "expected adjacency rows")));
    std::vector<std::vector<Py_ssize_t>> reference;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(refs.p); ++i) {
        reference.push_back(integers(PySequence_Fast_GET_ITEM(refs.p, i)));
        for (auto v : reference.back())
            if (!corr.contains(v))
                throw Unsupported();
    }
    Ref pairs_sequence(checked(PySequence_Fast(pairs_obj, "expected pairs")));
    std::vector<std::pair<Py_ssize_t, Py_ssize_t>> pairs;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(pairs_sequence.p); ++i) {
        auto pair = integers(PySequence_Fast_GET_ITEM(pairs_sequence.p, i));
        if (pair.size() != 2 || !corr.contains(pair[0]) || !corr.contains(pair[1]) ||
            pair[0] >= static_cast<Py_ssize_t>(reference.size()) ||
            pair[1] >= static_cast<Py_ssize_t>(reference.size()))
            throw Unsupported();
        pairs.emplace_back(pair[0], pair[1]);
    }
    std::unordered_map<Key, std::pair<double, bool>, Hash> seed;
    Py_ssize_t pos = 0;
    PyObject *seed_key, *seed_value;
    while (cache_enabled && PyDict_Next(seed_obj, &pos, &seed_key, &seed_value)) {
        if (!PyTuple_Check(seed_key) || PyTuple_GET_SIZE(seed_key) != 3 ||
            !PyTuple_Check(seed_value) || PyTuple_GET_SIZE(seed_value) != 2)
            throw Unsupported();
        Key key{integer(PyTuple_GET_ITEM(seed_key, 0)), integer(PyTuple_GET_ITEM(seed_key, 1)),
                integers(PyTuple_GET_ITEM(seed_key, 2))};
        double value = PyFloat_AsDouble(PyTuple_GET_ITEM(seed_value, 0));
        if (PyErr_Occurred())
            throw PythonError();
        int outer = PyObject_IsTrue(PyTuple_GET_ITEM(seed_value, 1));
        if (outer < 0)
            throw PythonError();
        seed.emplace(std::move(key), std::make_pair(value, outer != 0));
    }
    std::vector<State> states;
    states.reserve(pairs.size());
    std::deque<EventPtr> cit_events, outer_events;
    std::vector<Query> requests;
    requests.reserve(window);
    Ref errors(checked(PyList_New(0)));
    size_t requested = 0, computed = 0, cit_hits = 0, outer_hits = 0;
    size_t cacheable = 0, outer_cacheable = 0, compiled_queries = 0, python_queries = 0,
           windows = 0;
    bool has_errors = false;
    // All Python inputs are pinned and converted before this point. Keep the
    // GIL released for enumeration, low-order math and typed reductions; only
    // the bounded legacy numerical callback and object publication acquire it.
    ReleasedGIL unlocked;
    auto flush = [&]() {
        if (requests.empty())
            return;
        ++windows;
        std::vector<double> values(requests.size());
        bool fast = native_math && depth <= 1;
        if (fast) {
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto& q = requests[i];
                auto pair = pairs[q.edge];
                Py_ssize_t x = q.direction ? pair.second : pair.first;
                Py_ssize_t y = q.direction ? pair.first : pair.second;
                if (!low_order(corr, samples, x, y, q.key.condition, alpha, values[i])) {
                    fast = false;
                    break;
                }
            }
        }
        std::vector<bool> failures(requests.size(), false);
        if (fast)
            compiled_queries += requests.size();
        else {
            python_queries += requests.size();
            AcquiredGIL locked(unlocked);
            Ref queries(checked(PyList_New(requests.size())));
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto& q = requests[i];
                auto pair = pairs[q.edge];
                Key directed{q.direction ? pair.second : pair.first,
                             q.direction ? pair.first : pair.second, q.key.condition};
                auto query = key_object(directed);
                PyList_SET_ITEM(queries.p, i, query.release());
            }
            Ref raw(checked(PyObject_CallOneArg(callback, queries.p)));
            Ref returned(checked(PySequence_Fast(raw.p, "CI callback must return values")));
            if (PySequence_Fast_GET_SIZE(returned.p) != static_cast<Py_ssize_t>(requests.size())) {
                PyErr_SetString(PyExc_ValueError,
                                "CI callback returned a different number of values");
                throw PythonError();
            }
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto& q = requests[i];
                PyObject* item = PySequence_Fast_GET_ITEM(returned.p, i);
                if (PyExceptionInstance_Check(item)) {
                    auto pair = pairs[q.edge];
                    auto query =
                        key_object(Key{q.direction ? pair.second : pair.first,
                                       q.direction ? pair.first : pair.second, q.key.condition});
                    Ref failure(checked(PyTuple_Pack(2, query.p, item)));
                    if (PyList_Append(errors.p, failure.p) < 0)
                        throw PythonError();
                    failures[i] = true;
                    has_errors = true;
                    continue;
                }
                values[i] = PyFloat_AsDouble(item);
                if (PyErr_Occurred())
                    throw PythonError();
            }
        }
        computed += requests.size();
        for (size_t i = 0; i < requests.size(); ++i) {
            if (failures[i])
                continue;
            const auto& q = requests[i];
            double value = values[i];
            consume(states[q.edge], q.direction, q.key.condition, q.common, value, alpha, fci);
            if (q.event) {
                q.event->value = value;
                q.event->ready = true;
            }
            if (cache_enabled) {
                ++cacheable;
                outer_cacheable += outer_enabled;
            }
        }
        requests.clear();
    };
    for (size_t edge = 0; edge < pairs.size(); ++edge) {
        states.emplace_back();
        auto [a, b] = pairs[edge];
        std::vector<Py_ssize_t> pools[2];
        if (depth) {
            for (auto v : reference[a])
                if (v != b)
                    pools[0].push_back(v);
            for (auto v : reference[b])
                if (v != a)
                    pools[1].push_back(v);
        }
        std::unordered_set<Py_ssize_t> members[2] = {{pools[0].begin(), pools[0].end()},
                                                     {pools[1].begin(), pools[1].end()}};
        for (int direction = 0; direction < 2; ++direction) {
            const auto& pool = pools[direction];
            if (depth > static_cast<Py_ssize_t>(pool.size()))
                continue;
            std::vector<size_t> positions(depth);
            for (Py_ssize_t i = 0; i < depth; ++i)
                positions[i] = i;
            while (true) {
                std::vector<Py_ssize_t> condition;
                condition.reserve(depth);
                bool shared = true;
                for (auto p : positions) {
                    condition.push_back(pool[p]);
                    shared = shared && members[1 - direction].count(pool[p]);
                }
                ++requested;
                if (cache_enabled && direction && shared) {
                    if (outer_enabled)
                        ++outer_hits;
                    else
                        ++cit_hits;
                } else {
                    Key key{a, b, std::move(condition)};
                    auto cached = cache_enabled ? seed.find(key) : seed.end();
                    if (cached != seed.end()) {
                        auto [value, outer] = cached->second;
                        if (outer)
                            ++outer_hits;
                        else {
                            ++cit_hits;
                            if (outer_enabled) {
                                auto event = std::make_shared<Event>(Event{key, value, true, true});
                                append(outer_events, event, retain);
                                ++outer_cacheable;
                            }
                        }
                        consume(states[edge], direction, key.condition,
                                cache_enabled && !direction && shared, value, alpha, fci);
                    } else {
                        EventPtr event;
                        if (cache_enabled) {
                            event = std::make_shared<Event>(Event{key});
                            append(cit_events, event, retain);
                            if (outer_enabled)
                                append(outer_events, event, retain);
                        }
                        requests.push_back(Query{edge, direction,
                                                 cache_enabled && !direction && shared,
                                                 std::move(key), event});
                    }
                }
                if (requests.size() >= static_cast<size_t>(window))
                    flush();
                if (has_errors)
                    break;
                Py_ssize_t pivot = depth - 1;
                while (pivot >= 0 && positions[pivot] == pool.size() - depth + pivot)
                    --pivot;
                if (pivot < 0)
                    break;
                ++positions[pivot];
                for (Py_ssize_t i = pivot + 1; i < depth; ++i)
                    positions[i] = positions[i - 1] + 1;
            }
            if (has_errors)
                break;
        }
        if (has_errors)
            break;
    }
    flush();
    unlocked.acquire();
    Ref outcomes(checked(PyList_New(states.size())));
    for (size_t i = 0; i < states.size(); ++i) {
        auto& state = states[i];
        std::sort(state.separators.begin(), state.separators.end());
        Ref separators(checked(PyList_New(state.separators.size())));
        for (size_t j = 0; j < state.separators.size(); ++j) {
            auto condition = condition_object(state.separators[j].second);
            PyList_SET_ITEM(separators.p, j, condition.release());
        }
        auto members =
            condition_object(std::vector<Py_ssize_t>(state.members.begin(), state.members.end()));
        Ref maxima(checked(Py_BuildValue("[dd]", state.pmax[0], state.pmax[1])));
        PyList_SET_ITEM(
            outcomes.p, i,
            checked(Py_BuildValue("nnNNN", pairs[i].first, pairs[i].second, maxima.release(),
                                  separators.release(), members.release())));
    }
    auto updates = events_object(cit_events);
    auto outer_updates = events_object(outer_events);
    auto promotions = events_object(outer_events, true);
    Ref result(checked(PyDict_New()));
    count(result.p, "requests", requested);
    count(result.p, "computed", computed);
    count(result.p, "cit_hits", cit_hits);
    count(result.p, "outer_hits", outer_hits);
    count(result.p, "discarded", cacheable - PyList_GET_SIZE(updates.p));
    count(result.p, "outer_discarded", outer_cacheable - PyList_GET_SIZE(outer_updates.p));
    count(result.p, "compiled_low_order_queries", compiled_queries);
    count(result.p, "python_numeric_queries", python_queries);
    count(result.p, "native_windows", windows);
    put(result.p, "pairs", outcomes.release());
    put(result.p, "updates", updates.release());
    put(result.p, "outer_updates", outer_updates.release());
    put(result.p, "promotions", promotions.release());
    put(result.p, "errors", errors.release());
    return result.release();
}
PyObject* level(PyObject*, PyObject* args) {
    try {
        return level_impl(args);
    } catch (const Unsupported&) {
        Py_RETURN_NONE;
    } catch (const PythonError&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}
PyObject* batch_impl(PyObject* args) {
    PyObject *corr_obj, *queries_obj, *alpha_obj;
    Py_ssize_t samples;
    if (!PyArg_ParseTuple(args, "OnOO", &corr_obj, &samples, &queries_obj, &alpha_obj))
        return nullptr;
    Matrix corr(corr_obj);
    if (!corr.supported())
        throw Unsupported();
    double alpha = std::numeric_limits<double>::quiet_NaN();
    if (alpha_obj != Py_None) {
        alpha = PyFloat_AsDouble(alpha_obj);
        if (PyErr_Occurred())
            throw PythonError();
    }
    Ref queries(checked(PySequence_Fast(queries_obj, "expected CI queries")));
    auto size = PySequence_Fast_GET_SIZE(queries.p);
    std::vector<Key> typed_queries;
    typed_queries.reserve(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        Ref query(checked(
            PySequence_Fast(PySequence_Fast_GET_ITEM(queries.p, i), "expected a CI query")));
        if (PySequence_Fast_GET_SIZE(query.p) != 3)
            throw Unsupported();
        Py_ssize_t x = integer(PySequence_Fast_GET_ITEM(query.p, 0));
        Py_ssize_t y = integer(PySequence_Fast_GET_ITEM(query.p, 1));
        auto condition = integers(PySequence_Fast_GET_ITEM(query.p, 2));
        if (!corr.contains(x) || !corr.contains(y) || condition.size() > 1)
            throw Unsupported();
        for (auto v : condition)
            if (!corr.contains(v))
                throw Unsupported();
        typed_queries.push_back(Key{x, y, std::move(condition)});
    }
    std::vector<double> values(size);
    {
        ReleasedGIL unlocked;
        for (Py_ssize_t i = 0; i < size; ++i) {
            const auto& q = typed_queries[i];
            if (!low_order(corr, samples, q.a, q.b, q.condition, alpha, values[i]))
                throw Unsupported();
        }
    }
    Ref result(checked(PyList_New(size)));
    for (Py_ssize_t i = 0; i < size; ++i)
        PyList_SET_ITEM(result.p, i, checked(PyFloat_FromDouble(values[i])));
    return result.release();
}
PyObject* batch(PyObject*, PyObject* args) {
    try {
        return batch_impl(args);
    } catch (const Unsupported&) {
        Py_RETURN_NONE;
    } catch (const PythonError&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}
#include "_pdsep_native.hpp"

PyMethodDef methods[] = {
    {"level", level, METH_VARARGS, "Fused stable skeleton level; None for unsupported buffers."},
    {"batch", batch, METH_VARARGS,
     "Guarded typed low-order requests; None when the reference is required."},
    {"pdsep", pdsep, METH_VARARGS, "Fused bounded possible-d-separation search and reduction."},
    {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_fisher_native", nullptr, -1, methods};
} // namespace
PyMODINIT_FUNC PyInit__fisher_native() {
    PyObject* result = PyModule_Create(&module);
    if (!result)
        return nullptr;
    if (PyModule_AddIntConstant(result, "API_VERSION", 1) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
