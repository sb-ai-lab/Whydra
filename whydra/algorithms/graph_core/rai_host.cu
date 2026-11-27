// Host-side RAI accelerator: phase conditioning-set enumeration and the
// p-value lookup table. Pure host code (compiled through the CUDA extension
// plumbing so it shares the build); no device work.
//
// The reference replay enumerates, for every connected pair, the conditioning
// sets of order k drawn from the pair's two potential-parent pools, in Python
// `set` iteration order. That order is what the CI trace depends on, so it is
// never reproduced here: the *snapshot* enumeration order does not matter
// (the table is value-keyed), and the live replay passes the pools as Python
// tuples (== the set's own iteration order), so `combinations` over them is
// bit-identical to Python's `itertools.combinations` over the sets.
//
// Node ids are < n_vars, so a pair plus its (sorted) conditioning set packs
// into a single uint64 whenever `bits * (2 + k) <= 64`; otherwise the table
// falls back to byte-string keys. The packed path avoids ~1e6 allocations.
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace py = pybind11;

namespace {

inline int trailing_zeros(uint64_t bits) {
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward64(&index, bits);
    return static_cast<int>(index);
#else
    return __builtin_ctzll(bits);
#endif
}

inline int bits_for(int64_t n_vars) {
    int bits = 0;
    int64_t m = n_vars - 1;
    while (m > 0) {
        ++bits;
        m >>= 1;
    }
    return bits > 0 ? bits : 1;
}

inline std::string make_key(int32_t a, int32_t b, const int32_t* cs, int k) {
    std::string s;
    s.reserve(8 + 4 * static_cast<size_t>(k));
    s.append(reinterpret_cast<const char*>(&a), sizeof(a));
    s.append(reinterpret_cast<const char*>(&b), sizeof(b));
    if (k > 0) {
        s.append(reinterpret_cast<const char*>(cs), sizeof(int32_t) * static_cast<size_t>(k));
    }
    return s;
}

} // namespace

// Snapshot enumeration: `pairs` (M,2); `pa`/`pb` are M pools each (Python
// tuples in reference set order -- the order is irrelevant here).
// Returns (X, Y, S): flat int32 endpoints and a (N, order) int32 row of
// sorted conditioning sets, deduplicated per pair.
std::vector<torch::Tensor> enumerate_phase(torch::Tensor pairs,
                                           const std::vector<std::vector<int64_t>>& pa,
                                           const std::vector<std::vector<int64_t>>& pb,
                                           int64_t order, int64_t n_vars) {
    const int k = static_cast<int>(order);
    const int bits = bits_for(n_vars);
    const bool pack = static_cast<int64_t>(bits) * k <= 64;
    const auto M = static_cast<size_t>(pairs.size(0));
    TORCH_CHECK(pa.size() == M && pb.size() == M, "pool count mismatch");
    std::vector<int32_t> X, Y, S;
    X.reserve(M * 8);
    Y.reserve(M * 8);
    auto pr = pairs.accessor<int64_t, 2>();
    // Per-pair dedup of conditioning sets. The table is value-keyed, so an
    // exact dedup is only an optimisation; but the previous linear scan was
    // O(C^2) per pair (C = number of generated sets), which dominates at
    // p>=200 where a single phase can enumerate >2M sets. A hash set keeps it
    // O(C) while preserving exactly which keys survive (the first occurrence).
    std::unordered_set<uint64_t> seen_codes;
    std::unordered_set<std::string> seen_str;
    std::vector<int> idx;
    std::vector<int32_t> row;
    // k==1 dominates the order distribution (every search phase is order
    // 0 or 1): the conditioning set of a pair is just the deduped union of its
    // two pools, so skip the combination/sort machinery and dedup with a
    // per-pair node bitset.
    const int nwords = static_cast<int>((n_vars + 63) >> 6);
    std::vector<uint64_t> vbits;
    if (k == 1)
        vbits.assign(static_cast<size_t>(nwords), 0);
    for (size_t i = 0; i < M; ++i) {
        const int32_t a = static_cast<int32_t>(pr[i][0]);
        const int32_t b = static_cast<int32_t>(pr[i][1]);
        if (k == 1) {
            std::fill(vbits.begin(), vbits.end(), 0ull);
            const int64_t* pools1[2] = {pa[i].data(), pb[i].data()};
            const int sizes1[2] = {static_cast<int>(pa[i].size()), static_cast<int>(pb[i].size())};
            for (int which = 0; which < 2; ++which) {
                const int64_t* pl = pools1[which];
                const int n = sizes1[which];
                for (int j = 0; j < n; ++j) {
                    const int32_t v = static_cast<int32_t>(pl[j]);
                    uint64_t& w = vbits[static_cast<size_t>(v >> 6)];
                    const uint64_t m = 1ull << (v & 63);
                    if (w & m)
                        continue;
                    w |= m;
                    X.push_back(a);
                    Y.push_back(b);
                    S.push_back(v);
                }
            }
            continue;
        }
        if (pack)
            seen_codes.clear();
        else
            seen_str.clear();
        const int64_t* pools[2] = {pa[i].data(), pb[i].data()};
        const int sizes[2] = {static_cast<int>(pa[i].size()), static_cast<int>(pb[i].size())};
        if (k == 0) {
            X.push_back(a);
            Y.push_back(b);
            continue;
        }
        for (int which = 0; which < 2; ++which) {
            const int n = sizes[which];
            if (n < k)
                continue;
            idx.resize(static_cast<size_t>(k));
            for (int j = 0; j < k; ++j)
                idx[j] = j;
            row.resize(static_cast<size_t>(k));
            while (true) {
                for (int j = 0; j < k; ++j)
                    row[j] = static_cast<int32_t>(pools[which][idx[j]]);
                std::sort(row.begin(), row.end());
                bool dup = false;
                if (pack) {
                    uint64_t code = 0;
                    for (int j = 0; j < k; ++j)
                        code = (code << bits) | row[j];
                    dup = !seen_codes.insert(code).second;
                } else {
                    std::string sk(reinterpret_cast<const char*>(row.data()),
                                   sizeof(int32_t) * static_cast<size_t>(k));
                    dup = !seen_str.insert(std::move(sk)).second;
                }
                if (!dup) {
                    X.push_back(a);
                    Y.push_back(b);
                    for (int v : row)
                        S.push_back(v);
                }
                int j = k - 1;
                while (j >= 0 && idx[j] == n - k + j)
                    --j;
                if (j < 0)
                    break;
                ++idx[j];
                for (int t = j + 1; t < k; ++t)
                    idx[t] = idx[t - 1] + 1;
            }
        }
    }
    const int64_t N = static_cast<int64_t>(X.size());
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    auto Xt = torch::empty({N}, opts);
    auto Yt = torch::empty({N}, opts);
    auto St = torch::empty({N, k}, opts);
    if (N) {
        std::memcpy(Xt.data_ptr<int32_t>(), X.data(), sizeof(int32_t) * X.size());
        std::memcpy(Yt.data_ptr<int32_t>(), Y.data(), sizeof(int32_t) * Y.size());
        std::memcpy(St.data_ptr<int32_t>(), S.data(), sizeof(int32_t) * S.size());
    }
    return {Xt, Yt, St};
}

// Value-keyed p-value table for one phase.
class PValMap {
  public:
    PValMap(torch::Tensor x, torch::Tensor y, torch::Tensor S, torch::Tensor p, int64_t n_vars)
        : bits_(bits_for(n_vars)) {
        const int64_t N = x.numel();
        const int k = S.dim() == 2 ? static_cast<int>(S.size(1)) : 0;
        pack_ = static_cast<int64_t>(bits_) * (k + 2) <= 64;
        auto xa = x.accessor<int32_t, 1>();
        auto ya = y.accessor<int32_t, 1>();
        auto pa = p.accessor<double, 1>();
        const int32_t* Sp = S.numel() ? S.data_ptr<int32_t>() : nullptr;
        if (pack_) {
            // Flat open-addressing table: p-values are finite, so a NaN slot
            // is "empty". Avoids one heap node per entry (2.6M at p=200).
            size_t cap = 8;
            while (cap < static_cast<size_t>(N) * 3 / 2 + 1)
                cap <<= 1;
            fmask_ = cap - 1;
            fkeys_.assign(cap, 0);
            fvals_.assign(cap, std::numeric_limits<double>::quiet_NaN());
            for (int64_t i = 0; i < N; ++i) {
                uint64_t code = xa[i];
                code = (code << bits_) | static_cast<uint32_t>(ya[i]);
                for (int j = 0; j < k; ++j)
                    code =
                        (code << bits_) | static_cast<uint32_t>(Sp[static_cast<size_t>(i) * k + j]);
                size_t idx = hash_code(code);
                while (!std::isnan(fvals_[idx])) {
                    if (fkeys_[idx] == code)
                        break;
                    idx = (idx + 1) & fmask_;
                }
                fkeys_[idx] = code;
                fvals_[idx] = pa[i];
            }
        } else {
            map_.reserve(static_cast<size_t>(N) * 2);
            for (int64_t i = 0; i < N; ++i) {
                std::string key =
                    make_key(xa[i], ya[i], k ? Sp + static_cast<size_t>(i) * k : nullptr, k);
                map_.emplace(std::move(key), pa[i]);
            }
        }
    }

    // Missing -> NaN (p-values are finite in [0,1]).
    double get(int64_t a, int64_t b, const std::vector<int32_t>& cset) const {
        if (pack_) {
            uint64_t code = static_cast<uint64_t>(a);
            code = (code << bits_) | static_cast<uint32_t>(b);
            for (int32_t v : cset)
                code = (code << bits_) | static_cast<uint32_t>(v);
            return find_code(code);
        }
        std::string key = make_key(static_cast<int32_t>(a), static_cast<int32_t>(b), cset.data(),
                                   static_cast<int>(cset.size()));
        auto it = map_.find(key);
        if (it == map_.end())
            return std::numeric_limits<double>::quiet_NaN();
        return it->second;
    }

    inline double get_code(int32_t a, int32_t b, const int32_t* row, int k) const {
        uint64_t code = static_cast<uint64_t>(a);
        code = (code << bits_) | static_cast<uint32_t>(b);
        for (int i = 0; i < k; ++i)
            code = (code << bits_) | static_cast<uint32_t>(row[i]);
        return find_code(code);
    }

    // Live replay of one pair: enumerate the pair's conditioning sets in the
    // reference order (pools already in set-iteration order), sort/dedupe,
    // look up p, stop after the first p > alpha. Returns
    // (tested [(sorted_key, p)...], borderline) -- borderline is true if any
    // tested p is within `eps` of alpha, which the caller must re-check on the
    // CPU (rare); the last tested entry is the first hit.
    std::pair<std::vector<std::pair<std::vector<int32_t>, double>>, bool>
    decide(int64_t a, int64_t b, const std::vector<int64_t>& pa, const std::vector<int64_t>& pb,
           int64_t order, double alpha, double eps) const {
        const int k = static_cast<int>(order);
        std::vector<std::pair<std::vector<int32_t>, double>> out;
        bool borderline = false;
        auto record = [&](std::vector<int32_t>& row) -> bool {
            double p;
            if (pack_) {
                p = get_code(static_cast<int32_t>(a), static_cast<int32_t>(b), row.data(), k);
            } else {
                p = get(a, b, row);
            }
            if (p == p && std::fabs(p - alpha) <= eps)
                borderline = true;
            out.emplace_back(row, p);
            return p > alpha;
        };
        std::vector<std::vector<int32_t>> seen;
        std::vector<uint64_t> seen_codes;
        const int64_t* pools[2] = {pa.data(), pb.data()};
        const int sizes[2] = {static_cast<int>(pa.size()), static_cast<int>(pb.size())};
        for (int which = 0; which < 2; ++which) {
            const int n = sizes[which];
            if (k == 0) {
                if (which == 1)
                    break; // both pools collapse to the empty set
                std::vector<int32_t> empty;
                if (record(empty))
                    return {out, borderline};
                continue;
            }
            if (n < k)
                continue;
            std::vector<int> idx(static_cast<size_t>(k));
            for (int i = 0; i < k; ++i)
                idx[i] = i;
            std::vector<int32_t> row(static_cast<size_t>(k));
            while (true) {
                for (int i = 0; i < k; ++i)
                    row[i] = static_cast<int32_t>(pools[which][idx[i]]);
                std::sort(row.begin(), row.end());
                bool dup = false;
                if (pack_) {
                    uint64_t code = 0;
                    for (int i = 0; i < k; ++i)
                        code = (code << bits_) | row[i];
                    for (uint64_t c : seen_codes) {
                        if (c == code) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup)
                        seen_codes.push_back(code);
                } else {
                    for (auto& r : seen) {
                        if (r == row) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup)
                        seen.push_back(row);
                }
                if (!dup && record(row))
                    return {out, borderline};
                int i = k - 1;
                while (i >= 0 && idx[i] == n - k + i)
                    --i;
                if (i < 0)
                    break;
                ++idx[i];
                for (int t = i + 1; t < k; ++t)
                    idx[t] = idx[t - 1] + 1;
            }
        }
        return {out, borderline};
    }

    int64_t size() const {
        if (!pack_)
            return static_cast<int64_t>(map_.size());
        int64_t n = 0;
        for (double v : fvals_)
            if (!std::isnan(v))
                ++n;
        return n;
    }

    // Lookup by an already-sorted key vector (used by the native replay loop).
    double lookup(int64_t a, int64_t b, const std::vector<int32_t>& key) const {
        if (pack_) {
            return get_code(static_cast<int32_t>(a), static_cast<int32_t>(b), key.data(),
                            static_cast<int>(key.size()));
        }
        return get(a, b, key);
    }

  private:
    inline size_t hash_code(uint64_t code) const {
        code ^= code >> 33;
        code *= 0xff51afd7ed558ccdull;
        code ^= code >> 33;
        return static_cast<size_t>(code) & fmask_;
    }
    inline double find_code(uint64_t code) const {
        size_t idx = hash_code(code);
        while (!std::isnan(fvals_[idx])) {
            if (fkeys_[idx] == code)
                return fvals_[idx];
            idx = (idx + 1) & fmask_;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    int bits_;
    bool pack_ = false;
    size_t fmask_ = 0;
    std::vector<uint64_t> fkeys_;
    std::vector<double> fvals_;
    std::unordered_map<std::string, double> map_;
};

// ===========================================================================
// Native PDAG hot loops.
//
// These operate *directly* on the reference learner's Python `set` objects via
// the CPython C-API, so every iteration order is CPython's own set order (the
// order the reference would see at exactly that point) -- no order is ever
// reconstructed or assumed. The heavy decision work (connectivity bitsets,
// combination enumeration, p-value lookup) runs in C++; the graph itself stays
// the reference's own dict-of-sets, mutated with the same add/discard sequence
// the reference would use, so later Python phases still see identical sets.
//
// Adjacency is *fixed* across orientation (orient_edge only changes an edge's
// type, never its existence), so a single bitset is built once per call.
// ===========================================================================
namespace {

// bitset helpers taking explicit width
inline void bs_set(std::vector<uint64_t>& d, int w, int i, int j) {
    d[static_cast<size_t>(i) * w + (j >> 6)] |= 1ull << (j & 63);
}
inline void bs_clr(std::vector<uint64_t>& d, int w, int i, int j) {
    d[static_cast<size_t>(i) * w + (j >> 6)] &= ~(1ull << (j & 63));
}
inline bool bs_get(const std::vector<uint64_t>& d, int w, int i, int j) {
    return (d[static_cast<size_t>(i) * w + (j >> 6)] >> (j & 63)) & 1ull;
}

struct PDAGNat {
    int n;
    int w;
    std::vector<PyObject*> nk;
    std::vector<PyObject*> par;
    std::vector<PyObject*> und;
    std::vector<PyObject*> one; // cached {i} singleton sets
    std::vector<uint64_t> adj;  // any edge (i,j)
    std::vector<uint64_t> undb; // undirected edge (i,j)
    std::vector<uint64_t> parb; // parb(t,s) <=> s in parents(t) (s -> t)

    PDAGNat(PyObject* graph, int n_)
        : n(n_), w((n_ + 63) / 64), nk(n_), par(n_, nullptr), und(n_, nullptr), one(n_, nullptr),
          adj(static_cast<size_t>(w) * n_, 0), undb(static_cast<size_t>(w) * n_, 0),
          parb(static_cast<size_t>(w) * n_, 0) {
        for (int i = 0; i < n; ++i) {
            nk[i] = PyLong_FromLong(i);
            PyObject* inner = PyDict_GetItemWithError(graph, nk[i]);
            if (!inner) {
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_KeyError, "node");
                throw py::error_already_set();
            }
            PyObject* p = PyDict_GetItemString(inner, "<--");
            PyObject* u = PyDict_GetItemString(inner, "---");
            if (!p || !u)
                throw std::runtime_error("native PDAG: bad node entry");
            par[i] = p;
            und[i] = u;
            // ``{i}`` singleton, reused by the reference ``pot_parents``
            // difference so the replay never allocates a singleton set.
            one[i] = PySet_New(nullptr);
            if (!one[i] || PySet_Add(one[i], nk[i]) < 0)
                throw py::error_already_set();
        }
        for (int i = 0; i < n; ++i) {
            for_each(par[i], [&](int v) {
                bs_set(adj, w, i, v);
                bs_set(adj, w, v, i);  // adjacency is symmetric
                bs_set(parb, w, i, v); // v -> i
            });
            for_each(und[i], [&](int v) {
                bs_set(adj, w, i, v);
                bs_set(adj, w, v, i);
                bs_set(undb, w, i, v);
                bs_set(undb, w, v, i);
            });
        }
    }
    ~PDAGNat() {
        for (PyObject* o : nk)
            Py_XDECREF(o);
        for (PyObject* o : one)
            Py_XDECREF(o);
    }
    PDAGNat(const PDAGNat&) = delete;
    PDAGNat& operator=(const PDAGNat&) = delete;

    // The replay visits nodes in the order of the underlying Python set.
    template <typename F> static void for_each(PyObject* s, F&& f) {
        auto visit = [&](PyObject* key) {
            const long value = PyLong_AsLong(key);
            if (value == -1 && PyErr_Occurred())
                throw py::error_already_set();
            f(value);
        };
#if PY_VERSION_HEX < 0x030D0000
        // Older CPython versions expose an allocation-free scan of set entries.
        if (PySet_Check(s)) {
            Py_ssize_t pos = 0;
            PyObject* key = nullptr;
            Py_hash_t hash = 0;
            while (_PySet_NextEntry(s, &pos, &key, &hash))
                visit(key);
            if (PyErr_Occurred())
                throw py::error_already_set();
            return;
        }
#endif
        // Use the built-in set order even for a subclass overriding __iter__.
        auto it = py::reinterpret_steal<py::object>(PySet_Check(s) ? PySet_Type.tp_iter(s)
                                                                   : PyObject_GetIter(s));
        if (!it)
            throw py::error_already_set();
        while (auto item = py::reinterpret_steal<py::object>(PyIter_Next(it.ptr()))) {
            visit(item.ptr());
        }
        if (PyErr_Occurred())
            throw py::error_already_set();
    }

    static std::vector<int64_t> list_of(PyObject* s) {
        std::vector<int64_t> v;
        if (PySet_Check(s))
            v.reserve(static_cast<size_t>(PySet_Size(s)));
        for_each(s, [&](long value) { v.push_back(value); });
        return v;
    }

    inline bool adj_get(int i, int j) const { return bs_get(adj, w, i, j); }

    // orient src -> tgt; returns true if an undirected edge was flipped.
    // orient s -> t, mirroring ``_PDAG.orient_edge`` exactly. That primitive
    // adds s to parents(t) *unconditionally*, so orienting an edge whose
    // opposite direction is already directed produces a bidirected edge (a
    // real state change the later ``convert_bidirected`` undoes). parb(t,s)
    // tracks ``s in parents(t)`` so the no-op case is detected.
    inline bool orient(int s, int t) {
        bool changed = false;
        if (bs_get(undb, w, s, t)) {
            bs_clr(undb, w, s, t);
            bs_clr(undb, w, t, s);
            if (PySet_Discard(und[s], nk[t]) < 0)
                throw py::error_already_set();
            if (PySet_Discard(und[t], nk[s]) < 0)
                throw py::error_already_set();
            changed = true;
        }
        if (!bs_get(parb, w, t, s)) {
            if (PySet_Add(par[t], nk[s]) < 0)
                throw py::error_already_set();
            bs_set(parb, w, t, s);
            changed = true;
        }
        return changed;
    }

    inline void delete_edge(int i, int j) {
        if (PySet_Discard(par[j], nk[i]) < 0)
            throw py::error_already_set();
        if (PySet_Discard(par[i], nk[j]) < 0)
            throw py::error_already_set();
        if (PySet_Discard(und[i], nk[j]) < 0)
            throw py::error_already_set();
        if (PySet_Discard(und[j], nk[i]) < 0)
            throw py::error_already_set();
        bs_clr(adj, w, i, j);
        bs_clr(adj, w, j, i);
        bs_clr(undb, w, i, j);
        bs_clr(undb, w, j, i);
        bs_clr(parb, w, j, i);
        bs_clr(parb, w, i, j);
    }
};

// Build the 3-D sepset membership set from the reference's ``_sepset`` dict:
// (min(i,j), max(i,j), z) for every z in the pair's separator.
std::unordered_set<uint64_t> build_sep(PyObject* sepset_dict, int n) {
    std::unordered_set<uint64_t> S;
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(sepset_dict, &pos, &key, &val)) {
        PyObject* k0 = PyTuple_GetItem(key, 0);
        PyObject* k1 = PyTuple_GetItem(key, 1);
        long a = PyLong_AsLong(k0);
        long b = PyLong_AsLong(k1);
        long lo = a < b ? a : b, hi = a < b ? b : a;
        uint64_t base = static_cast<uint64_t>(lo) * n + hi;
        for (int64_t z : PDAGNat::list_of(val))
            S.insert((base * n) + z);
    }
    return S;
}
inline bool sep_has(const std::unordered_set<uint64_t>& S, int n, int x, int y, int z) {
    long lo = x < y ? x : y, hi = x < y ? y : x;
    return S.find((static_cast<uint64_t>(lo) * n + hi) * n + z) != S.end();
}

// makes a fresh Python set == ``s.copy()`` (inserts in s's iteration order).
inline PyObject* py_copy(PyObject* s) {
    PyObject* c = PySet_New(s);
    if (!c)
        throw py::error_already_set();
    return c;
}

} // namespace

namespace {

// ---- orient_v_structures -------------------------------------------------
void do_orient_v(PDAGNat& G, const std::unordered_set<uint64_t>& S, int n,
                 const std::vector<int64_t>& en) {
    std::vector<std::vector<int64_t>> pre_nb(n);
    for (int64_t z : en) {
        PyObject* c = py_copy(G.und[z]);
        pre_nb[z] = PDAGNat::list_of(c);
        Py_DECREF(c);
    }
    for (int64_t z : en) {
        const auto& xy = pre_nb[z];
        for (size_t i = 0; i < xy.size(); ++i) {
            for (size_t j = i + 1; j < xy.size(); ++j) {
                int x = static_cast<int>(xy[i]), y = static_cast<int>(xy[j]);
                if (G.adj_get(x, y))
                    continue;
                if (sep_has(S, n, x, y, static_cast<int>(z)))
                    continue;
                G.orient(x, static_cast<int>(z));
                G.orient(y, static_cast<int>(z));
            }
        }
    }
    std::vector<std::vector<int64_t>> nb2(n), par2(n);
    for (int64_t z : en) {
        PyObject* cn = py_copy(G.und[z]);
        nb2[z] = PDAGNat::list_of(cn);
        Py_DECREF(cn);
        PyObject* cp = py_copy(G.par[z]);
        par2[z] = PDAGNat::list_of(cp);
        Py_DECREF(cp);
    }
    for (int64_t z : en) {
        const auto& nbs = nb2[z];
        const auto& pars = par2[z];
        for (int64_t xv : nbs) {
            int x = static_cast<int>(xv);
            for (int64_t yv : pars) {
                int y = static_cast<int>(yv);
                if (G.adj_get(x, y))
                    continue;
                if (sep_has(S, n, x, y, static_cast<int>(z)))
                    continue;
                G.orient(x, static_cast<int>(z));
            }
        }
    }
}

void orient_v_native(py::object graph_, py::object en_nodes_, py::object sepset_dict_, int n) {
    PDAGNat G(graph_.ptr(), n);
    auto S = build_sep(sepset_dict_.ptr(), n);
    do_orient_v(G, S, n, PDAGNat::list_of(en_nodes_.ptr()));
}

// ---- Meek rules ----------------------------------------------------------
// The rules below test *existence* of a witness (the reference orients x -> y
// once, on the first witness it finds), so they never need the witness set
// itself -- only whether it is non-empty. Every such test is expressed as an
// AND of two rows of the PDAG bitsets (parb/undb/adj), which is O(w) and
// allocation-free. Iteration *order* of x (the undirected neighbours of y) is
// kept exactly (copied from und[y] in CPython order), and the live rows
// parb(y,.), parb(x,.) are re-read per x, so behaviour matches the reference
// rule-for-rule; only the per-check set allocations disappear.

// AND of a row of `a` (i) with a row of a DIFFERENT array `b` (j), non-zero?
inline bool rows_cross_and_nonzero(const std::vector<uint64_t>& a, int w, int i,
                                   const std::vector<uint64_t>& b, int j) {
    const uint64_t* pa = &a[static_cast<size_t>(i) * w];
    const uint64_t* pb = &b[static_cast<size_t>(j) * w];
    uint64_t acc = 0;
    for (int t = 0; t < w; ++t)
        acc |= pa[t] & pb[t];
    return acc != 0;
}

// row(i) of `a` AND NOT row(j) of `b`, non-zero? (cross-array)
inline bool rows_cross_andnot_nonzero(const std::vector<uint64_t>& a, int w, int i,
                                      const std::vector<uint64_t>& b, int j) {
    const uint64_t* pa = &a[static_cast<size_t>(i) * w];
    const uint64_t* pb = &b[static_cast<size_t>(j) * w];
    uint64_t acc = 0;
    for (int t = 0; t < w; ++t)
        acc |= pa[t] & ~pb[t];
    return acc != 0;
}
template <typename F>
inline void for_row_bits(const std::vector<uint64_t>& d, int w, int i, F&& f) {
    const uint64_t* r = &d[static_cast<size_t>(i) * w];
    for (int t = 0; t < w; ++t) {
        uint64_t x = r[t];
        while (x) {
            int b = trailing_zeros(x);
            x &= x - 1;
            f(t * 64 + b);
        }
    }
}

bool rule1(PDAGNat& G, const std::vector<int64_t>& en) {
    bool mod = false;
    for (int64_t yv : en) {
        int y = static_cast<int>(yv);
        if (PySet_Size(G.und[y]) == 0)
            continue;
        PyObject* c = py_copy(G.und[y]);
        auto xs = PDAGNat::list_of(c);
        Py_DECREF(c);
        for (int64_t xv : xs) {
            int x = static_cast<int>(xv);
            // exists z in par(x) with z not adjacent to y
            if (rows_cross_andnot_nonzero(G.parb, G.w, x, G.adj, y)) {
                G.orient(x, y);
                mod = true;
            }
        }
    }
    return mod;
}

bool rule2(PDAGNat& G, const std::vector<int64_t>& en) {
    bool mod = false;
    for (int64_t yv : en) {
        int y = static_cast<int>(yv);
        if (PySet_Size(G.und[y]) == 0 || PySet_Size(G.par[y]) == 0)
            continue;
        PyObject* c = py_copy(G.und[y]);
        auto xs = PDAGNat::list_of(c);
        Py_DECREF(c);
        for (int64_t xv : xs) {
            int x = static_cast<int>(xv);
            // exists z in par(y) with x in par(z)
            bool hit = false;
            for_row_bits(G.parb, G.w, y, [&](int z) {
                if (!hit && bs_get(G.parb, G.w, z, x))
                    hit = true;
            });
            if (hit) {
                G.orient(x, y);
                mod = true;
            }
        }
    }
    return mod;
}

bool rule3(PDAGNat& G, const std::vector<int64_t>& en) {
    bool mod = false;
    for (int64_t yv : en) {
        int y = static_cast<int>(yv);
        if (PySet_Size(G.und[y]) == 0 || PySet_Size(G.par[y]) == 0)
            continue;
        PyObject* c = py_copy(G.und[y]);
        auto xs = PDAGNat::list_of(c);
        Py_DECREF(c);
        for (int64_t xv : xs) {
            int x = static_cast<int>(xv);
            // candidates = par(y) & und(x); orient if two are non-adjacent.
            // parb(y,.) is re-read here (par(y) grows as earlier x's orient).
            std::vector<int> cand;
            const uint64_t* p = &G.parb[static_cast<size_t>(y) * G.w];
            const uint64_t* u = &G.undb[static_cast<size_t>(x) * G.w];
            for (int t = 0; t < G.w; ++t) {
                uint64_t z = p[t] & u[t];
                while (z) {
                    int b = trailing_zeros(z);
                    z &= z - 1;
                    cand.push_back(t * 64 + b);
                }
            }
            bool hit = false;
            for (size_t i = 0; i < cand.size() && !hit; ++i)
                for (size_t j = i + 1; j < cand.size(); ++j)
                    if (!G.adj_get(cand[i], cand[j])) {
                        hit = true;
                        break;
                    }
            if (hit) {
                G.orient(x, y);
                mod = true;
            }
        }
    }
    return mod;
}

bool rule4(PDAGNat& G, const std::vector<int64_t>& en) {
    bool mod = false;
    for (int64_t yv : en) {
        int y = static_cast<int>(yv);
        if (PySet_Size(G.und[y]) == 0 || PySet_Size(G.par[y]) == 0)
            continue;
        PyObject* c = py_copy(G.und[y]);
        auto xs = PDAGNat::list_of(c);
        Py_DECREF(c);
        for (int64_t xv : xs) {
            int x = static_cast<int>(xv);
            // exists z in par(y): adj(z,x) and par(z) & und(x) != empty
            bool hit = false;
            for_row_bits(G.parb, G.w, y, [&](int z) {
                if (hit || !G.adj_get(z, x))
                    return;
                if (rows_cross_and_nonzero(G.parb, G.w, z, G.undb, x))
                    hit = true;
            });
            if (hit) {
                G.orient(x, y);
                mod = true;
            }
        }
    }
    return mod;
}

void do_maxorient(PDAGNat& G, const std::vector<int64_t>& rs, const std::vector<int64_t>& en) {
    bool modified = true;
    while (modified) {
        modified = false;
        for (int64_t r : rs) {
            bool m = false;
            switch (r) {
            case 1:
                m = rule1(G, en);
                break;
            case 2:
                m = rule2(G, en);
                break;
            case 3:
                m = rule3(G, en);
                break;
            case 4:
                m = rule4(G, en);
                break;
            default:
                break;
            }
            modified = modified || m;
        }
    }
}

void maxorient_native(py::object graph_, py::object rules_, py::object en_nodes_, int n) {
    PDAGNat G(graph_.ptr(), n);
    do_maxorient(G, PDAGNat::list_of(rules_.ptr()), PDAGNat::list_of(en_nodes_.ptr()));
}

void do_convert(PDAGNat& G, const std::vector<int64_t>& ns) {
    for (size_t i = 0; i < ns.size(); ++i) {
        for (size_t j = i + 1; j < ns.size(); ++j) {
            int x = static_cast<int>(ns[i]), y = static_cast<int>(ns[j]);
            int xy = PySet_Contains(G.par[x], G.nk[y]);
            if (xy < 0)
                throw py::error_already_set();
            if (!xy)
                continue;
            int yx = PySet_Contains(G.par[y], G.nk[x]);
            if (yx < 0)
                throw py::error_already_set();
            if (!yx)
                continue;
            if (PySet_Discard(G.par[x], G.nk[y]) < 0)
                throw py::error_already_set();
            if (PySet_Discard(G.par[y], G.nk[x]) < 0)
                throw py::error_already_set();
            if (PySet_Add(G.und[x], G.nk[y]) < 0)
                throw py::error_already_set();
            if (PySet_Add(G.und[y], G.nk[x]) < 0)
                throw py::error_already_set();
            bs_set(G.undb, G.w, x, y);
            bs_set(G.undb, G.w, y, x);
            bs_clr(G.parb, G.w, x, y);
            bs_clr(G.parb, G.w, y, x);
            bs_set(G.adj, G.w, x, y);
            bs_set(G.adj, G.w, y, x);
        }
    }
}

void convert_bidirected_native(py::object graph_, py::object nodes_, int n) {
    PDAGNat G(graph_.ptr(), n);
    do_convert(G, PDAGNat::list_of(nodes_.ptr()));
}

// One native call for the whole reference ``maximally_orient_edges`` sequence:
// orient_v_structures(en) -> maximally_orient_pattern(rules, all) ->
// convert_bidirected_to_undirected(en). Shares one PDAGNat construction and one
// sepset index across all three steps.
void maximally_orient_edges_native(py::object graph_, py::object en_nodes_, py::object sepset_dict_,
                                   py::object all_nodes_, py::object rules_, int n) {
    PDAGNat G(graph_.ptr(), n);
    auto S = build_sep(sepset_dict_.ptr(), n);
    std::vector<int64_t> en = PDAGNat::list_of(en_nodes_.ptr());
    do_orient_v(G, S, n, en);
    do_maxorient(G, PDAGNat::list_of(rules_.ptr()), PDAGNat::list_of(all_nodes_.ptr()));
    do_convert(G, en);
}

// Build the phase's snapshot enumeration directly from the graph: for every
// connected pair, pool_x = parents(x) | undirected(x) minus the other endpoint.
// Only the *set* of conditioning sets matters (the p-value table is value
// keyed), so the pool order need not match Python's; duplicates are removed by
// enumerate_phase's per-pair dedupe. Disconnected pairs are dropped, matching
// the reference's job assembly.
std::vector<torch::Tensor> enumerate_phase_graph(torch::Tensor pairs, py::object graph_,
                                                 int64_t order, int64_t n_vars) {
    PDAGNat G(graph_.ptr(), static_cast<int>(n_vars));
    const auto M = static_cast<size_t>(pairs.size(0));
    auto pr = pairs.accessor<int64_t, 2>();
    std::vector<int64_t> fpair;
    std::vector<std::vector<int64_t>> pa, pb;
    fpair.reserve(M * 2);
    for (size_t i = 0; i < M; ++i) {
        int a = static_cast<int>(pr[i][0]);
        int b = static_cast<int>(pr[i][1]);
        if (!G.adj_get(a, b))
            continue;
        fpair.push_back(a);
        fpair.push_back(b);
        pa.emplace_back();
        pb.emplace_back();
        // Order 0 has a single conditioning set (the empty one), so the pools
        // are never read by ``enumerate_phase``. Building them is pure waste:
        // at p=100 the first phase has all C(100,2)=4950 pairs, and iterating
        // par/und of both endpoints through the C-API cost ~16 ms there.
        if (order == 0)
            continue;
        auto& va = pa.back();
        auto& vb = pb.back();
        PDAGNat::for_each(G.par[a], [&](int v) {
            if (v != b)
                va.push_back(v);
        });
        PDAGNat::for_each(G.und[a], [&](int v) {
            if (v != b)
                va.push_back(v);
        });
        PDAGNat::for_each(G.par[b], [&](int v) {
            if (v != a)
                vb.push_back(v);
        });
        PDAGNat::for_each(G.und[b], [&](int v) {
            if (v != a)
                vb.push_back(v);
        });
    }
    const int64_t np = static_cast<int64_t>(pa.size());
    torch::Tensor pt;
    if (np) {
        pt = torch::from_blob(fpair.data(), {np, 2}, torch::kInt64).clone();
    } else {
        pt = torch::empty({0, 2}, torch::kInt64);
    }
    return enumerate_phase(pt, pa, pb, order, n_vars);
}

py::list adjacency_native(py::object graph_, int n) {
    PDAGNat G(graph_.ptr(), n);
    py::list out(n);
    for (int i = 0; i < n; ++i) {
        py::object val = py::int_(0);
        py::object shift_amt = py::int_(64);
        for (int wi = G.w - 1; wi >= 0; --wi) {
            uint64_t word = 0;
            for (int b = 0; b < 64; ++b) {
                int j = wi * 64 + b;
                if (j < n && bs_get(G.adj, G.w, i, j))
                    word |= (1ull << b);
            }
            py::object w = py::int_(static_cast<unsigned long long>(word));
            PyObject* sh = PyNumber_Lshift(val.ptr(), shift_amt.ptr());
            if (!sh)
                throw py::error_already_set();
            PyObject* ored = PyNumber_Or(sh, w.ptr());
            Py_DECREF(sh);
            if (!ored)
                throw py::error_already_set();
            val = py::reinterpret_steal<py::object>(ored);
        }
        out[i] = val;
    }
    return out;
}

} // namespace

// ---- refine_pairs replay -------------------------------------------------
namespace {

// Reference pool ``pot_parents`` = ``(parents(node) | undirected(node)) - {other}``
// is built inline in the replay below with ``PyNumber_Subtract(get_pu(node),
// one[other])``: ``PyNumber_Or``/``PyNumber_Subtract`` are the C slots behind
// ``set.union``/``set.__sub__`` (identical result *and* iteration order), the
// union is cached per node, and the ``{other}`` singleton is cached on the PDAG.

py::tuple refine_pairs_native(py::object graph, py::object pairs, int64_t order, py::object sepset,
                              py::object pmap, double alpha, double eps, py::object trace,
                              py::object cpu_cb) {
    PyObject* g = graph.ptr();
    Py_ssize_t np = PyDict_Size(g); // node count = keys of the graph dict
    PDAGNat G(g, static_cast<int>(np));
    PValMap* M = pmap.is_none() ? nullptr : pmap.cast<PValMap*>();
    PyObject* sepset_dict = sepset.ptr();
    PyObject* tr = trace.is_none() ? nullptr : trace.ptr();
    PyObject* cb = cpu_cb.is_none() ? nullptr : cpu_cb.ptr();
    const int k = static_cast<int>(order);

    long n_ci = 0, n_sync = 0, n_border = 0;

    PyObject* fast = PySequence_Fast(pairs.ptr(), "pairs");
    if (!fast)
        throw py::error_already_set();
    Py_ssize_t npairs = PySequence_Fast_GET_SIZE(fast);

    // Cached ``par[node] | und[node]`` per node. A fresh union has the same
    // iteration order as the reference's per-pair ``parents | undirected``, so
    // as long as nothing touched the node we can reuse it and skip one set
    // allocation per pair. ``delete_edge`` (the only mutation inside a phase)
    // changes par/und of exactly its two endpoints, so both cache slots are
    // cleared there.
    std::vector<PyObject*> pu(static_cast<size_t>(G.n), nullptr);
    auto get_pu = [&](int node) -> PyObject* {
        if (!pu[node]) {
            pu[node] = PyNumber_Or(G.par[node], G.und[node]);
            if (!pu[node])
                throw py::error_already_set();
        }
        return pu[node];
    };

    // Process one conditioning set (already sorted): lookup, boundary/CPU
    // re-check, trace, delete+sepset. Returns true if it was independent and
    // the pair's loop must stop. ``a``/``b`` captured by reference.
    auto process = [&](long a, long b, const std::vector<int32_t>& key) -> bool {
        ++n_ci;
        double p = std::numeric_limits<double>::quiet_NaN();
        if (M)
            p = M->lookup(a, b, key);
        bool need_cpu = std::isnan(p);
        bool border = !need_cpu && std::fabs(p - alpha) <= eps;
        if (need_cpu || border) {
            if (M)
                ++n_sync;
            if (border)
                ++n_border;
            if (cb) {
                PyObject* pk = PyTuple_New(static_cast<Py_ssize_t>(key.size()));
                for (size_t i = 0; i < key.size(); ++i)
                    PyTuple_SET_ITEM(pk, i, PyLong_FromLong(key[i]));
                PyObject* r = PyObject_CallFunctionObjArgs(cb, G.nk[a], G.nk[b], pk, nullptr);
                Py_DECREF(pk);
                if (!r)
                    throw py::error_already_set();
                p = PyFloat_AsDouble(r);
                Py_DECREF(r);
            }
        }
        bool independent = p > alpha;
        if (tr) {
            PyObject* kt = PyTuple_New(static_cast<Py_ssize_t>(key.size()));
            for (size_t i = 0; i < key.size(); ++i)
                PyTuple_SET_ITEM(kt, i, PyLong_FromLong(key[i]));
            PyObject* ent = Py_BuildValue("(llNi)", a, b, kt, independent ? 1 : 0);
            PyList_Append(tr, ent);
            Py_DECREF(ent);
        }
        if (independent) {
            G.delete_edge(static_cast<int>(a), static_cast<int>(b));
            Py_CLEAR(pu[a]);
            Py_CLEAR(pu[b]);
            PyObject* hs = PySet_New(nullptr);
            for (int32_t v : key)
                if (PySet_Add(hs, G.nk[v]) < 0) {
                    Py_DECREF(hs);
                    throw py::error_already_set();
                }
            PyObject* hk = Py_BuildValue("(ll)", a < b ? a : b, a < b ? b : a);
            PyDict_SetItem(sepset_dict, hk, hs);
            Py_DECREF(hs);
            Py_DECREF(hk);
        }
        return independent;
    };

    std::vector<int> idx;
    std::vector<int32_t> key;
    std::vector<uint64_t> vbits;
    if (k == 1)
        vbits.assign(static_cast<size_t>((G.n + 63) >> 6), 0);
    // Dedup of conditioning sets across the two pools: a packed 64-bit code
    // (one byte per node id) when representable, else a byte string.
    const bool pack_dup = (k >= 1 && k <= 8 && G.n <= 256);
    std::unordered_set<uint64_t> seen64;
    std::unordered_set<std::string> seen;

    for (Py_ssize_t pi = 0; pi < npairs; ++pi) {
        PyObject* pr = PySequence_Fast_GET_ITEM(fast, pi);
        long a = PyLong_AsLong(PyTuple_GetItem(pr, 0));
        long b = PyLong_AsLong(PyTuple_GetItem(pr, 1));
        if (a < 0 || b < 0) {
            Py_DECREF(fast);
            throw py::error_already_set();
        }
        if (!G.adj_get(static_cast<int>(a), static_cast<int>(b)))
            continue;

        if (k == 0) {
            // Order 0: the only conditioning set is the empty one; the pools
            // are irrelevant, so skip ``pot_parents`` entirely.
            key.clear();
            process(a, b, key);
            continue;
        }
        if (k == 1) {
            // Order 1: the conditioning sets are the elements of
            // ``pot_parents(a,b)`` followed by ``pot_parents(b,a)``. Both pools
            // are built with the exact reference op ``(par|und) - {other}`` --
            // the cached union is the reference's own ``parents | undirected``
            // (invalidated on deletion), and ``PyNumber_Subtract`` is the C
            // slot behind ``set.__sub__``. Iterating the *difference* set
            // reproduces the reference order; FILTERING ``other`` out of the
            // union while iterating is NOT equivalent (CPython rebuilds the
            // difference set's table, which reorders elements once the set
            // outgrows the 8-slot minimum). Both pools are built up front:
            // ``process`` may delete the edge midway and mutate the live sets.
            // A per-pair node bitset dedupes across the two pools.
            std::fill(vbits.begin(), vbits.end(), 0ull);
            key.resize(1);
            PyObject* ua =
                PyNumber_Subtract(get_pu(static_cast<int>(a)), G.one[static_cast<int>(b)]);
            if (!ua) {
                Py_DECREF(fast);
                throw py::error_already_set();
            }
            PyObject* ub =
                PyNumber_Subtract(get_pu(static_cast<int>(b)), G.one[static_cast<int>(a)]);
            if (!ub) {
                Py_DECREF(ua);
                Py_DECREF(fast);
                throw py::error_already_set();
            }
            const std::vector<int64_t> pa1 = PDAGNat::list_of(ua);
            const std::vector<int64_t> pb1 = PDAGNat::list_of(ub);
            Py_DECREF(ua);
            Py_DECREF(ub);
            bool stop1 = false;
            for (const std::vector<int64_t>* pl : {&pa1, &pb1}) {
                for (int64_t vv : *pl) {
                    const int32_t v = static_cast<int32_t>(vv);
                    uint64_t& w = vbits[static_cast<size_t>(v >> 6)];
                    const uint64_t m = 1ull << (v & 63);
                    if (w & m)
                        continue;
                    w |= m;
                    key[0] = v;
                    if (process(a, b, key)) {
                        stop1 = true;
                        break;
                    }
                }
                if (stop1)
                    break;
            }
            continue;
        }

        PyObject* pA = PyNumber_Subtract(get_pu(static_cast<int>(a)), G.one[static_cast<int>(b)]);
        if (!pA) {
            Py_DECREF(fast);
            throw py::error_already_set();
        }
        PyObject* pB = PyNumber_Subtract(get_pu(static_cast<int>(b)), G.one[static_cast<int>(a)]);
        if (!pB) {
            Py_DECREF(pA);
            Py_DECREF(fast);
            throw py::error_already_set();
        }
        std::vector<int64_t> pa = PDAGNat::list_of(pA);
        std::vector<int64_t> pb = PDAGNat::list_of(pB);
        Py_DECREF(pA);
        Py_DECREF(pB);

        if (pack_dup)
            seen64.clear();
        else
            seen.clear();
        bool stop = false;
        for (int which = 0; which < 2 && !stop; ++which) {
            const std::vector<int64_t>& pool = which ? pb : pa;
            int m = static_cast<int>(pool.size());
            if (m < k)
                continue;
            idx.resize(k);
            for (int i = 0; i < k; ++i)
                idx[i] = i;
            for (;;) {
                key.resize(k);
                for (int i = 0; i < k; ++i)
                    key[i] = static_cast<int32_t>(pool[idx[i]]);
                std::sort(key.begin(), key.end());
                bool fresh;
                if (pack_dup) {
                    uint64_t code = 0;
                    for (int i = 0; i < k; ++i)
                        code |= static_cast<uint64_t>(key[i]) << (8 * i);
                    fresh = seen64.insert(code).second;
                } else {
                    std::string sk(reinterpret_cast<const char*>(key.data()), sizeof(int32_t) * k);
                    fresh = seen.insert(sk).second;
                }
                if (fresh) {
                    if (process(a, b, key)) {
                        stop = true;
                        break;
                    }
                }
                int i = k - 1;
                while (i >= 0 && idx[i] == m - k + i)
                    --i;
                if (i < 0)
                    break;
                ++idx[i];
                for (int t = i + 1; t < k; ++t)
                    idx[t] = idx[t - 1] + 1;
            }
        }
    }
    Py_DECREF(fast);
    for (PyObject* o : pu)
        Py_XDECREF(o);
    return py::make_tuple(n_ci, n_sync, n_border);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("enumerate_phase", &enumerate_phase);
    m.def("enumerate_phase_graph", &enumerate_phase_graph);
    m.def("orient_v_structures_native", &orient_v_native);
    m.def("maximally_orient_pattern_native", &maxorient_native);
    m.def("convert_bidirected_native", &convert_bidirected_native);
    m.def("maximally_orient_edges_native", &maximally_orient_edges_native);
    m.def("adjacency_native", &adjacency_native);
    m.def("refine_pairs_native", &refine_pairs_native);
    py::class_<PValMap>(m, "PValMap")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t>())
        .def("get", &PValMap::get)
        .def("decide", &PValMap::decide)
        .def("lookup", &PValMap::lookup)
        .def("size", &PValMap::size);
}
