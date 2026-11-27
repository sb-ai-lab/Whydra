// Included inside _fisher_native.cpp's anonymous namespace after its helpers.
// Portable C++17; Python objects are accessed only while the GIL is acquired.
struct PdsepValue {
    std::vector<Py_ssize_t> original;
    Key key;
    double value = std::numeric_limits<double>::quiet_NaN();
    PyObject* error = nullptr; // Borrowed from the invocation's owned errors list.
    bool missing = true;
    Py_ssize_t origin = -1;
};

struct PdsepState {
    Py_ssize_t index = 0, x = 0, y = 0, cap = 0, retain = 0;
    Py_ssize_t current_order = 1, reported_order = 0;
    double alpha = .05, pmax = -std::numeric_limits<double>::infinity();
    std::vector<Py_ssize_t> tf, diff, separator;
    std::unordered_set<Py_ssize_t> adjacent;
    std::unordered_map<Key, std::pair<double, Py_ssize_t>, Hash> cached;
    using Update = std::pair<Key, double>;
    std::list<Update> updates;
    std::unordered_map<Key, std::list<Update>::iterator, Hash> update_index;
    std::vector<size_t> positions;
    std::vector<PdsepValue> window;
    size_t singleton = 0, window_size = 32;
    size_t consumed = 0, requests = 0, computed = 0, outer_hits = 0, cit_hits = 0;
    size_t discarded_cit = 0, discarded_outer = 0, windows = 0;
    bool initialized = false, available = false, done = false, separated = false;
    PyObject* error = nullptr;

    bool next_condition(std::vector<Py_ssize_t>& condition) {
        if (current_order == 1) {
            if (singleton == diff.size())
                return false;
            condition.assign(1, diff[singleton++]);
            return true;
        }
        while (available) {
            condition.clear();
            bool inside = true;
            for (size_t position : positions) {
                condition.push_back(tf[position]);
                inside = inside && adjacent.count(tf[position]) != 0;
            }
            size_t pivot = positions.size();
            while (pivot && positions[pivot - 1] == tf.size() - positions.size() + pivot - 1)
                --pivot;
            if (!pivot)
                available = false;
            else {
                ++positions[pivot - 1];
                for (size_t i = pivot; i < positions.size(); ++i)
                    positions[i] = positions[i - 1] + 1;
            }
            if (!inside)
                return true;
        }
        return false;
    }

    bool prepare() {
        window.clear();
        while (current_order <= cap) {
            reported_order = current_order;
            if (!initialized) {
                singleton = 0;
                positions.clear();
                available = current_order > 1 && static_cast<size_t>(current_order) <= tf.size();
                if (available) {
                    positions.resize(static_cast<size_t>(current_order));
                    for (size_t i = 0; i < positions.size(); ++i)
                        positions[i] = i;
                }
                initialized = true;
                window_size = 32;
            }
            std::vector<Py_ssize_t> condition;
            while (window.size() < window_size && next_condition(condition)) {
                PdsepValue item;
                item.original = condition;
                std::sort(condition.begin(), condition.end());
                item.key = Key{x, y, condition};
                auto hit = cached.find(item.key);
                if (hit != cached.end()) {
                    item.missing = false;
                    item.value = hit->second.first;
                    item.origin = hit->second.second;
                    if (item.origin == 0)
                        ++outer_hits;
                    else
                        ++cit_hits;
                }
                window.push_back(std::move(item));
            }
            if (!window.empty())
                return true;
            if (current_order == cap)
                break;
            // Empty impossible orders still update the same final reported
            // order as range(start_order, cap + 1), without integer overflow.
            if (current_order >= static_cast<Py_ssize_t>(tf.size()) && current_order >= 1) {
                reported_order = cap;
                break;
            }
            ++current_order;
            initialized = false;
        }
        done = true;
        return false;
    }

    void consume_window() {
        ++windows;
        requests += window.size();
        for (const auto& item : window) {
            computed += item.missing;
            // Speculative successes enter the bounded tail before the ordered
            // prefix is consumed. Exceptions, including unconsumed ones, do not.
            if (retain > 0 && !item.error && (item.missing || item.origin == 1)) {
                auto previous = update_index.find(item.key);
                if (previous != update_index.end()) {
                    previous->second->second = item.value;
                    updates.splice(updates.end(), updates, previous->second);
                } else {
                    updates.emplace_back(item.key, item.value);
                    auto last = updates.end();
                    --last;
                    update_index.emplace(item.key, last);
                }
                if (updates.size() > static_cast<size_t>(retain)) {
                    const auto& discarded = updates.front().first;
                    ++discarded_outer;
                    if (cached.find(discarded) == cached.end())
                        ++discarded_cit;
                    update_index.erase(discarded);
                    updates.pop_front();
                }
            }
        }
        for (const auto& item : window) {
            ++consumed;
            if (item.error) {
                error = item.error;
                done = true;
                return;
            }
            if (item.value > pmax)
                pmax = item.value;
            if (item.value >= alpha) {
                separator = item.original;
                separated = true;
                done = true;
                return;
            }
        }
        window_size = std::min<size_t>(2048, window_size * 4);
    }
};

std::unique_ptr<PdsepState> pdsep_parse(PyObject* descriptor, const Matrix& corr) {
    Ref row(checked(PySequence_Fast(descriptor, "expected indexed PDSEP descriptor")));
    if (PySequence_Fast_GET_SIZE(row.p) != 2)
        throw Unsupported();
    auto state = std::make_unique<PdsepState>();
    state->index = integer(PySequence_Fast_GET_ITEM(row.p, 0));
    Ref args(checked(
        PySequence_Fast(PySequence_Fast_GET_ITEM(row.p, 1), "expected PDSEP pair arguments")));
    auto length = PySequence_Fast_GET_SIZE(args.p);
    if (length != 10 && length != 11)
        throw Unsupported();
    auto item = [&](Py_ssize_t index) { return PySequence_Fast_GET_ITEM(args.p, index); };
    state->x = integer(item(0));
    state->y = integer(item(1));
    state->tf = integers(item(2));
    state->diff = integers(item(3));
    auto adjacent = integers(item(4));
    state->adjacent.insert(adjacent.begin(), adjacent.end());
    state->cap = integer(item(5));
    state->alpha = PyFloat_AsDouble(item(6));
    if (PyErr_Occurred())
        throw PythonError();
    state->pmax = PyFloat_AsDouble(item(7));
    if (PyErr_Occurred())
        throw PythonError();
    state->retain = integer(item(9));
    if (length == 11)
        state->current_order = integer(item(10));
    if (state->cap < 0 || state->retain < 0 || state->current_order < 0 ||
        !corr.contains(state->x) || !corr.contains(state->y) || !PyDict_Check(item(8)))
        throw Unsupported();
    for (auto node : state->tf)
        if (!corr.contains(node))
            throw Unsupported();
    for (auto node : state->diff)
        if (!corr.contains(node))
            throw Unsupported();
    // The typed seed is immutable throughout the search. New numerical values
    // only enter the publication tail, matching the Python generator exactly.
    Py_ssize_t position = 0;
    PyObject *key_obj, *value_obj;
    while (PyDict_Next(item(8), &position, &key_obj, &value_obj)) {
        if (value_obj == Py_None)
            continue;
        Ref value(checked(PySequence_Fast(value_obj, "expected cached value and origin")));
        if (PySequence_Fast_GET_SIZE(value.p) != 2)
            throw Unsupported();
        Key key{state->x, state->y, integers(key_obj)};
        double probability = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(value.p, 0));
        if (PyErr_Occurred())
            throw PythonError();
        Py_ssize_t origin = integer(PySequence_Fast_GET_ITEM(value.p, 1));
        state->cached.emplace(std::move(key), std::make_pair(probability, origin));
    }
    return state;
}

Ref pdsep_result(const PdsepState& state) {
    Ref result(checked(PyDict_New()));
    put(result.p, "x", PyLong_FromSsize_t(state.x));
    put(result.p, "y", PyLong_FromSsize_t(state.y));
    put(result.p, "order", PyLong_FromSsize_t(state.reported_order));
    put(result.p, "pmax", PyFloat_FromDouble(state.pmax));
    count(result.p, "consumed", state.consumed);
    count(result.p, "requests", state.requests);
    count(result.p, "computed", state.computed);
    count(result.p, "outer_hits", state.outer_hits);
    count(result.p, "cit_hits", state.cit_hits);
    count(result.p, "discarded_cit_insertions", state.discarded_cit);
    count(result.p, "discarded_outer_insertions", state.discarded_outer);
    Ref separator(state.separated ? condition_object(state.separator).release()
                                  : Py_NewRef(Py_None));
    put(result.p, "separator", separator.release());
    put(result.p, "error", Py_NewRef(state.error ? state.error : Py_None));
    Ref updates(checked(PyList_New(state.updates.size())));
    Py_ssize_t position = 0;
    for (const auto& update : state.updates) {
        auto condition = condition_object(update.first.condition);
        PyList_SET_ITEM(updates.p, position++,
                        checked(Py_BuildValue("Nd", condition.release(), update.second)));
    }
    put(result.p, "updates", updates.release());
    return result;
}

PyObject* pdsep_impl(PyObject* args) {
    PyObject *corr_obj, *descriptors_obj, *callback;
    Py_ssize_t samples;
    if (!PyArg_ParseTuple(args, "OnOO", &corr_obj, &samples, &descriptors_obj, &callback))
        return nullptr;
    Matrix corr(corr_obj);
    if (!corr.supported() || !PyCallable_Check(callback))
        throw Unsupported();
    Ref descriptors(checked(PySequence_Fast(descriptors_obj, "expected PDSEP descriptors")));
    std::vector<std::unique_ptr<PdsepState>> states;
    auto descriptor_count = PySequence_Fast_GET_SIZE(descriptors.p);
    states.reserve(static_cast<size_t>(descriptor_count));
    for (Py_ssize_t i = 0; i < descriptor_count; ++i)
        states.push_back(pdsep_parse(PySequence_Fast_GET_ITEM(descriptors.p, i), corr));
    // This owner is declared before ReleasedGIL: exception unwinding restores
    // the GIL before any Python exception instances are decref'ed.
    Ref exception_owners(checked(PyList_New(0)));
    Ref consumed_errors(checked(PyList_New(0)));
    std::vector<size_t> active, completed;
    active.reserve(4);
    completed.reserve(states.size());
    size_t remaining = 0;
    ReleasedGIL unlocked;
    while (true) {
        while (active.size() < 4 && remaining < states.size()) {
            size_t index = remaining++;
            if (states[index]->prepare())
                active.push_back(index);
            else
                completed.push_back(index);
        }
        if (active.empty())
            break;
        std::vector<PdsepValue*> missing;
        for (size_t index : active)
            for (auto& value : states[index]->window)
                if (value.missing)
                    missing.push_back(&value);
        {
            AcquiredGIL locked(unlocked);
            if (PyErr_CheckSignals() < 0)
                throw PythonError();
            Ref queries(checked(PyList_New(missing.size())));
            for (size_t i = 0; i < missing.size(); ++i) {
                auto query = key_object(missing[i]->key);
                PyList_SET_ITEM(queries.p, i, query.release());
            }
            // Keep this callback even for an empty missing list. The Python
            // generator performs the same send/barrier for fully cached windows.
            Ref raw(checked(PyObject_CallOneArg(callback, queries.p)));
            Ref returned(checked(PySequence_Fast(raw.p, "CI callback must return values")));
            if (PySequence_Fast_GET_SIZE(returned.p) != static_cast<Py_ssize_t>(missing.size())) {
                PyErr_SetString(PyExc_ValueError,
                                "CI callback returned a different number of values");
                throw PythonError();
            }
            for (size_t i = 0; i < missing.size(); ++i) {
                PyObject* value = PySequence_Fast_GET_ITEM(returned.p, i);
                if (PyExceptionInstance_Check(value)) {
                    if (PyList_Append(exception_owners.p, value) < 0)
                        throw PythonError();
                    missing[i]->error = value;
                } else {
                    missing[i]->value = PyFloat_AsDouble(value);
                    if (PyErr_Occurred())
                        throw PythonError();
                }
            }
        }
        // Pure C++: ordered reduction, bounded LRU publication and generation
        // of each surviving pair's next window all execute without the GIL.
        std::vector<size_t> following;
        following.reserve(4);
        for (size_t index : active) {
            auto& state = *states[index];
            state.consume_window();
            if (!state.done && state.prepare())
                following.push_back(index);
            else
                completed.push_back(index);
        }
        {
            AcquiredGIL locked(unlocked);
            for (size_t index : active) {
                auto& state = *states[index];
                if (state.done && state.error && PyList_Append(consumed_errors.p, state.error) < 0)
                    throw PythonError();
            }
            // Retain only the exception actually returned for each failed
            // pair. Speculative error objects are bounded by this one round.
            if (PyList_SetSlice(exception_owners.p, 0, PyList_GET_SIZE(exception_owners.p),
                                nullptr) < 0)
                throw PythonError();
        }
        active.swap(following);
    }
    unlocked.acquire();
    Ref results(checked(PyList_New(completed.size())));
    for (size_t position = 0; position < completed.size(); ++position) {
        const auto& state = *states[completed[position]];
        auto result = pdsep_result(state);
        PyList_SET_ITEM(results.p, position,
                        checked(Py_BuildValue("nN", state.index, result.release())));
    }
    return results.release();
}

PyObject* pdsep(PyObject*, PyObject* args) {
    try {
        return pdsep_impl(args);
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
