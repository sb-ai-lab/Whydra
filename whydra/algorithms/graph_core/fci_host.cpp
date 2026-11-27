// Host-side helpers for the Possible-D-SEP stage of FCI.
//
// The Python driver used to build every conditioning-set block with a full
// C(|PDS|, k) gather and filter. This enumerator generates only the subsets
// that pass the reference filter (not entirely in adj(a) and not entirely in
// adj(b)), so forbidden blocks are never materialized. Sets are classified by
// the first selected element outside adj(a) and the first outside adj(b); the
// three orderings of those witnesses give a disjoint partition.
//
// Collider endpoints are accumulated from unordered pairs of neighbors.
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void require_sepset_csr_cpu(const torch::Tensor& S, const torch::Tensor& ranges, int p,
                            const char* stage) {
    TORCH_CHECK(S.is_cpu() && S.scalar_type() == torch::kInt64 && S.is_contiguous(), stage,
                ": S must be contiguous int64 CPU");
    TORCH_CHECK(S.dim() == 1, stage, ": S must be 1-D, got dim ", S.dim());
    TORCH_CHECK(ranges.is_cpu() && ranges.scalar_type() == torch::kInt32 && ranges.is_contiguous(),
                stage, ": sepset ranges must be contiguous int32 CPU");
    TORCH_CHECK(ranges.dim() == 3, stage, ": sepset ranges must have shape (p,p,2), got dim ",
                ranges.dim());
    TORCH_CHECK(ranges.size(0) == p && ranges.size(1) == p && ranges.size(2) == 2, stage,
                ": sepset ranges must have shape (", p, ",", p, ",2), got (", ranges.size(0), ",",
                ranges.size(1), ",", ranges.size(2), ")");
    if (p == 0) {
        return;
    }
    const int32_t* R = ranges.data_ptr<int32_t>();
    const int64_t n_ranges = ranges.numel() / 2;
    int32_t min_start = R[0];
    int32_t min_end = R[1];
    int32_t max_end = R[1];
    for (int64_t i = 0; i < n_ranges; ++i) {
        const int32_t start = R[2 * i];
        const int32_t end = R[2 * i + 1];
        min_start = std::min(min_start, start);
        min_end = std::min(min_end, end);
        max_end = std::max(max_end, end);
        TORCH_CHECK(!(start == -1 && end != -1), stage,
                    ": a missing separator must be stored as (-1,-1)");
        TORCH_CHECK(!(start >= 0 && end < start), stage,
                    ": sepset range end must not precede start");
    }
    TORCH_CHECK(min_start >= -1, stage, ": sepset range start must be >= -1, got ", min_start);
    TORCH_CHECK(min_end >= -1, stage, ": sepset range end must be >= -1, got ", min_end);
    TORCH_CHECK(max_end <= S.numel(), stage, ": sepset ranges point past S (max end ", max_end,
                " > S.numel() ", S.numel(), ")");
}

void gen_combos(const std::vector<int>& src, int choose, int start, std::vector<int>& cur,
                std::vector<int>& out) {
    if (choose == 0) {
        out.insert(out.end(), cur.begin(), cur.end());
        return;
    }
    const int n = static_cast<int>(src.size());
    for (int i = start; i <= n - choose; ++i) {
        cur.push_back(src[i]);
        gen_combos(src, choose - 1, i + 1, cur, out);
        cur.pop_back();
    }
}

// Emit pre^p × mand × mid^m × suf^s, in sorted element order.
bool emit_product(const std::vector<int>& pre, const std::vector<int>& mid,
                  const std::vector<int>& suf, const std::vector<int>& mand, int p, int m, int s,
                  size_t cap, std::vector<std::vector<int>>& out) {
    std::vector<int> cur, flat_pre, flat_mid, flat_suf;
    gen_combos(pre, p, 0, cur, flat_pre);
    cur.clear();
    gen_combos(mid, m, 0, cur, flat_mid);
    cur.clear();
    gen_combos(suf, s, 0, cur, flat_suf);
    const size_t n_pre = p == 0 ? 1 : flat_pre.size() / p;
    const size_t n_mid = m == 0 ? 1 : flat_mid.size() / m;
    const size_t n_suf = s == 0 ? 1 : flat_suf.size() / s;
    const size_t total = n_pre * n_mid * n_suf;
    if (out.size() + total > cap) {
        return false;
    }
    for (size_t a = 0; a < n_pre; ++a) {
        for (size_t b = 0; b < n_mid; ++b) {
            for (size_t c = 0; c < n_suf; ++c) {
                std::vector<int> one;
                one.reserve(static_cast<size_t>(p + m + s + mand.size()));
                for (int t = 0; t < p; ++t) {
                    one.push_back(flat_pre[a * p + t]);
                }
                if (!mand.empty()) {
                    one.push_back(mand[0]);
                }
                for (int t = 0; t < m; ++t) {
                    one.push_back(flat_mid[b * m + t]);
                }
                for (size_t t = 1; t < mand.size(); ++t) {
                    one.push_back(mand[t]);
                }
                for (int t = 0; t < s; ++t) {
                    one.push_back(flat_suf[c * s + t]);
                }
                out.push_back(std::move(one));
            }
        }
    }
    return true;
}

std::vector<int> filter_between(const std::vector<int>& pool, int lo, int hi,
                                const std::vector<uint8_t>& keep) {
    std::vector<int> out;
    for (int i = lo; i < hi; ++i) {
        if (keep[pool[i]]) {
            out.push_back(pool[i]);
        }
    }
    return out;
}

std::vector<int> tail(const std::vector<int>& pool, int from) {
    return std::vector<int>(pool.begin() + from, pool.end());
}

// Append all k-subsets of pool that are neither entirely in A nor entirely in
// B. Returns false if the count would exceed cap.
bool append_allowed(const std::vector<int>& pool, const std::vector<uint8_t>& in_a,
                    const std::vector<uint8_t>& in_b, int k, size_t cap,
                    std::vector<std::vector<int>>& out) {
    const int n = static_cast<int>(pool.size());
    if (k < 2 || k > n) {
        return true;
    }
    int max_v = 0;
    for (int v : pool) {
        max_v = std::max(max_v, v);
    }
    std::vector<uint8_t> both(static_cast<size_t>(max_v + 1), 0);
    for (int v : pool) {
        both[v] = static_cast<uint8_t>(in_a[v] && in_b[v]);
    }

    // i < j: first non-A is i (in B\A), first non-B is j.
    for (int i = 0; i < n; ++i) {
        if (in_a[pool[i]] || !in_b[pool[i]]) {
            continue;
        }
        const std::vector<int> pre = filter_between(pool, 0, i, both);
        for (int j = i + 1; j < n; ++j) {
            if (in_b[pool[j]]) {
                continue;
            }
            const std::vector<int> mid = filter_between(pool, i + 1, j, in_b);
            const std::vector<int> suf = tail(pool, j + 1);
            for (int a = 0; a <= std::min<int>(k - 2, pre.size()); ++a) {
                for (int b = 0; b <= std::min<int>(k - 2 - a, mid.size()); ++b) {
                    const int c = k - 2 - a - b;
                    if (c > static_cast<int>(suf.size())) {
                        continue;
                    }
                    if (!emit_product(pre, mid, suf, {pool[i], pool[j]}, a, b, c, cap, out)) {
                        return false;
                    }
                }
            }
        }
    }

    // j < i: first non-B is j (in A\B), first non-A is i.
    for (int j = 0; j < n; ++j) {
        if (in_b[pool[j]] || !in_a[pool[j]]) {
            continue;
        }
        const std::vector<int> pre = filter_between(pool, 0, j, both);
        for (int i = j + 1; i < n; ++i) {
            if (in_a[pool[i]]) {
                continue;
            }
            const std::vector<int> mid = filter_between(pool, j + 1, i, in_a);
            const std::vector<int> suf = tail(pool, i + 1);
            for (int a = 0; a <= std::min<int>(k - 2, pre.size()); ++a) {
                for (int b = 0; b <= std::min<int>(k - 2 - a, mid.size()); ++b) {
                    const int c = k - 2 - a - b;
                    if (c > static_cast<int>(suf.size())) {
                        continue;
                    }
                    if (!emit_product(pre, mid, suf, {pool[j], pool[i]}, a, b, c, cap, out)) {
                        return false;
                    }
                }
            }
        }
    }

    // i == j: first selected element outside A ∪ B.
    for (int i = 0; i < n; ++i) {
        if (in_a[pool[i]] || in_b[pool[i]]) {
            continue;
        }
        const std::vector<int> pre = filter_between(pool, 0, i, both);
        const std::vector<int> suf = tail(pool, i + 1);
        const std::vector<int> none;
        for (int a = 0; a <= std::min<int>(k - 1, pre.size()); ++a) {
            const int c = k - 1 - a;
            if (c > static_cast<int>(suf.size())) {
                continue;
            }
            if (!emit_product(pre, none, suf, {pool[i]}, a, 0, c, cap, out)) {
                return false;
            }
        }
    }
    return true;
}

// Sparse graph index shared by all queries on the snapshot. Each discovered
// vertex is evaluated once, after the vertices preceding it in breadth order.
class PdsSolver {
  public:
    explicit PdsSolver(torch::Tensor amat)
        : n_(static_cast<int>(amat.size(0))), marks_(amat.data_ptr<int32_t>()), neighbors_(n_),
          successors_(n_), directed_(n_), labels_(n_) {
        for (int v = 0; v < n_; ++v) {
            labels_[v] = "X" + std::to_string(v + 1);
            for (int u = 0; u < n_; ++u) {
                if (!mark(v, u))
                    continue;
                neighbors_[v].push_back(u);
                if (mark(u, v) == 2 || mark(u, v) == -1)
                    successors_[v].push_back(u);
            }
        }
    }

    std::vector<int> possible_dsep(int source, int excluded) {
        std::vector<int> order, result;
        std::vector<uint8_t> seen(n_), processed(n_);
        seen[source] = seen[excluded] = 1;
        auto discover = [&](int u) {
            if (!seen[u]) {
                seen[u] = 1;
                order.push_back(u);
            }
        };
        for (int u : neighbors_[source])
            discover(u);
        for (size_t i = 0; i < order.size(); ++i) {
            const int v = order[i];
            bool accept = mark(source, v) != 0;
            // A potential predecessor supplies evidence only after it has
            // been expanded. Semidirected closure is snapshot-invariant.
            if (!accept) {
                for (int u : neighbors_[v]) {
                    if (processed[u] && (reaches(u, source) || reaches(u, v))) {
                        accept = true;
                        break;
                    }
                }
            }
            if (accept)
                result.push_back(v);
            processed[v] = 1;
            if (mark(source, v) == 1) {
                for (int u : neighbors_[v]) {
                    if (mark(u, v) == 1)
                        discover(u);
                }
            }
        }
        std::sort(result.begin(), result.end(),
                  [&](int u, int v) { return labels_[u] > labels_[v]; });
        return result;
    }

  private:
    int mark(int u, int v) const { return marks_[static_cast<int64_t>(u) * n_ + v]; }

    bool reaches(int src, int dest) {
        auto& closure = directed_[src];
        if (closure.empty()) {
            closure.assign(n_, 0);
            std::vector<int> stack{src};
            // Positive-length reachability: a source reaches itself only on a cycle.
            while (!stack.empty()) {
                const int v = stack.back();
                stack.pop_back();
                for (int u : successors_[v]) {
                    if (!closure[u]) {
                        closure[u] = 1;
                        stack.push_back(u);
                    }
                }
            }
        }
        return closure[dest] != 0;
    }

    int n_;
    const int32_t* marks_;
    std::vector<std::vector<int>> neighbors_, successors_;
    std::vector<std::vector<uint8_t>> directed_;
    std::vector<std::string> labels_;
};

} // namespace

// Returns (ok, sizes, flat, offsets). flat[offsets[i] : offsets[i+1]] is the
// row-major block for sizes[i], ordered lexicographically by position in PDS.
// Exceeding cap returns ok=false with empty vectors. A successful search with
// no blocks returns ok=true and offsets={0}.
std::tuple<bool, std::vector<int>, std::vector<int>, std::vector<int64_t>>
pds_blocks(int64_t p, std::vector<int> pds, std::vector<int> adj_a, std::vector<int> adj_b,
           int64_t m_max, int64_t cap) {
    for (int v : pds) {
        TORCH_CHECK(v >= 0 && v < p, "pds_blocks: pds value ", v, " outside [0,p) with p=", p);
    }
    for (int v : adj_a) {
        TORCH_CHECK(v >= 0 && v < p, "pds_blocks: adj_a value ", v, " outside [0,p) with p=", p);
    }
    for (int v : adj_b) {
        TORCH_CHECK(v >= 0 && v < p, "pds_blocks: adj_b value ", v, " outside [0,p) with p=", p);
    }
    const int n = static_cast<int>(pds.size());
    std::vector<uint8_t> in_a(static_cast<size_t>(p), 0);
    std::vector<uint8_t> in_b(static_cast<size_t>(p), 0);
    for (int v : adj_a) {
        if (v >= 0 && v < p) {
            in_a[v] = 1;
        }
    }
    for (int v : adj_b) {
        if (v >= 0 && v < p) {
            in_b[v] = 1;
        }
    }
    std::unordered_map<int, int> pos;
    for (int i = 0; i < n; ++i) {
        pos[pds[i]] = i;
    }

    const int upper = m_max < 0 ? n : std::min<int64_t>(n, m_max);
    std::vector<int> sizes;
    std::vector<int> flat;
    std::vector<int64_t> offsets;
    size_t total = 0;

    for (int k = 2; k <= upper; ++k) {
        std::vector<std::vector<int>> combos;
        if (!append_allowed(pds, in_a, in_b, k, static_cast<size_t>(cap), combos)) {
            return {false, {}, {}, {}};
        }
        if (combos.empty()) {
            continue;
        }
        std::vector<std::pair<std::vector<int>, std::vector<int>>> keyed;
        keyed.reserve(combos.size());
        for (auto& c : combos) {
            if (static_cast<int>(c.size()) != k) {
                return {false, {}, {}, {}};
            }
            std::vector<int> key;
            key.reserve(c.size());
            for (int v : c) {
                auto it = pos.find(v);
                if (it == pos.end()) {
                    return {false, {}, {}, {}};
                }
                key.push_back(it->second);
            }
            keyed.emplace_back(std::move(key), std::move(c));
        }
        std::sort(keyed.begin(), keyed.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        total += keyed.size();
        if (total > static_cast<size_t>(cap)) {
            return {false, {}, {}, {}};
        }
        sizes.push_back(k);
        offsets.push_back(static_cast<int64_t>(flat.size()));
        for (const auto& kv : keyed) {
            flat.insert(flat.end(), kv.second.begin(), kv.second.end());
        }
    }
    offsets.push_back(static_cast<int64_t>(flat.size()));
    return {true, sizes, flat, offsets};
}

// Reference v-structure pass on the Possible-D-SEP graph, in place.
int64_t pds_colliders(torch::Tensor amat, torch::Tensor sepset_S, torch::Tensor sepset_ranges) {
    TORCH_CHECK(amat.is_cpu() && amat.scalar_type() == torch::kInt32, "amat int32 CPU");
    TORCH_CHECK(amat.dim() == 2 && amat.is_contiguous(), "amat must be contiguous (p,p)");
    TORCH_CHECK(amat.size(0) == amat.size(1), "amat must be square");
    const int p = static_cast<int>(amat.size(0));
    require_sepset_csr_cpu(sepset_S, sepset_ranges, p, "pds_colliders");
    const int32_t* R = sepset_ranges.data_ptr<int32_t>();
    const int64_t* S = sepset_S.data_ptr<int64_t>();
    int32_t* A = amat.data_ptr<int32_t>();

    auto has_sep = [&](int x, int z, int y) {
        const int64_t base = (static_cast<int64_t>(x) * p + z) * 2;
        const int a = R[base], b = R[base + 1];
        if (a < 0 || b < a) {
            return false;
        }
        for (int i = a; i < b; ++i) {
            if (S[i] == y) {
                return true;
            }
        }
        return false;
    };

    // A missing edge between the outer vertices and a separating set that
    // omits the center imply two arrowheads (FCI R0). Candidate orientations
    // share a center and commute. Count both directions on symmetric skeletons.
    auto edge = [&](int u, int v) -> int32_t& { return A[static_cast<int64_t>(u) * p + v]; };
    int64_t n_orient = 0;
    for (int center = 0; center < p; ++center) {
        std::vector<int> adjacent;
        for (int v = 0; v < p; ++v) {
            if (edge(v, center) || edge(center, v))
                adjacent.push_back(v);
        }
        auto unshielded = [&](int u, int v) {
            return edge(u, center) && edge(center, v) && !edge(u, v);
        };
        std::vector<uint8_t> orient(adjacent.size());
        for (size_t right = 0; right < adjacent.size(); ++right) {
            const int v = adjacent[right];
            for (size_t left = 0; left < right; ++left) {
                const int u = adjacent[left];
                const int directions = unshielded(u, v) + unshielded(v, u);
                if (directions && !has_sep(u, v, center) && !has_sep(v, u, center)) {
                    orient[left] = orient[right] = 1;
                    n_orient += directions;
                }
            }
        }
        for (size_t i = 0; i < adjacent.size(); ++i) {
            if (orient[i])
                edge(adjacent[i], center) = 1;
        }
    }
    return n_orient;
}

std::vector<std::vector<int>> pds_batch(torch::Tensor amat,
                                        std::vector<std::pair<int, int>> pairs) {
    TORCH_CHECK(amat.is_cpu() && amat.scalar_type() == torch::kInt32, "amat int32 CPU");
    TORCH_CHECK(amat.dim() == 2 && amat.is_contiguous(), "amat must be contiguous (p,p)");
    TORCH_CHECK(amat.size(0) == amat.size(1), "amat must be square");
    const int64_t p = amat.size(0);
    for (const auto& pr : pairs) {
        TORCH_CHECK(pr.first >= 0 && pr.first < p && pr.second >= 0 && pr.second < p,
                    "pds_batch: pair (", pr.first, ",", pr.second, ") is outside [0,p) with p=", p);
    }
    PdsSolver solver(amat);
    std::vector<std::vector<int>> out;
    out.reserve(pairs.size());
    for (const auto& pr : pairs) {
        out.push_back(solver.possible_dsep(pr.first, pr.second));
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pds_batch", &pds_batch, pybind11::arg("amat"), pybind11::arg("pairs"));
    m.def("pds_blocks", &pds_blocks, pybind11::arg("p"), pybind11::arg("pds"),
          pybind11::arg("adj_a"), pybind11::arg("adj_b"), pybind11::arg("m_max"),
          pybind11::arg("cap"));
    m.def("pds_colliders", &pds_colliders, pybind11::arg("amat"), pybind11::arg("sepset_S"),
          pybind11::arg("sepset_ranges"));
    m.def("preload", []() {});
}
