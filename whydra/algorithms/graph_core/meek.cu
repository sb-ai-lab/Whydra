// Meek R1–R3 on a dense endpoint matrix (int8).
//
// G[i, j] uses Endpoint values: 0 none, -1 tail, +1 arrow.
// i → j is G[i,j]==ARROW and G[j,i]==TAIL. Undirected is both tails.
// Adjacency (nonzero) is frozen; only endpoints change.
//
// Compact neighbors is a multi-CTA kernel; R1–R3 stay one CTA. Outer loops
// over vertices match sequential find_unshielded_triples / find_triangles /
// find_kites. R1 packs unshielded V-shapes (i,k) at center j once
// (multi-CTA count/scan/fill, CSR v_off[j]); apply is parallel inside j,
// with a CTA sync after each j. R2 packs 3-cliques i<j<k once with a
// multi-CTA count/scan/fill into t_i/t_j/t_k (t_off is the pack CSR).
// Apply is a 256-slot SMEM window over that list. Each slot holds a batch
// index; a hit atomicMins that index (not tid). The thread whose g equals
// the min hit paints from registers (no SMEM bounce to thread 0). Slots
// with g <= hit — or the whole window on a miss — pull the next items so
// all occupied slots re-test after an orientation. R3 packs kites
// (i,j,k,l) once in find_kites order, then the same batch-index window:
// the thread whose g equals the min hit paints i→l from registers,
// suffix re-tests. Apply is
// immediate (Gauss–Seidel) so the CPDAG matches Python Meek.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_check.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include "cuda_touch.cuh"
#include "cuda_validate.cuh"

namespace {

constexpr int kTail = -1;
constexpr int kArrow = 1;
constexpr int kBlock = 256;
constexpr int kRuleR1 = 1;
constexpr int kRuleR2 = 2;
constexpr int kRuleR3 = 4;

__device__ __forceinline__ int8_t at(const int8_t* G, int p, int i, int j) {
    return G[static_cast<long long>(i) * p + j];
}

__device__ __forceinline__ void put(int8_t* G, int p, int i, int j, int8_t v) {
    G[static_cast<long long>(i) * p + j] = v;
}

__device__ __forceinline__ bool is_adj(const int8_t* G, int p, int i, int j) {
    return at(G, p, i, j) != 0;
}

__device__ __forceinline__ bool is_dir(const int8_t* G, int p, int i, int j) {
    return at(G, p, i, j) == kArrow && at(G, p, j, i) == kTail;
}

__device__ __forceinline__ bool is_und(const int8_t* G, int p, int i, int j) {
    return at(G, p, i, j) == kTail && at(G, p, j, i) == kTail;
}

__device__ __forceinline__ bool is_bi(const int8_t* G, int p, int i, int j) {
    return at(G, p, i, j) == kArrow && at(G, p, j, i) == kArrow;
}

// True iff a directed path src -> ... -> dst exists over fully directed
// edges (arrow at (u,v), tail at (v,u)).  Mirrors CausalGraph.is_ancestor_of:
// a node is never its own ancestor, and bidirected/undirected edges do not
// extend the path.  Used by the Meek cycle guard (causal-learn parity):
// orienting src -> tgt is skipped when tgt is already an ancestor of src.
__device__ __forceinline__ bool reaches_directed(const int8_t* G, int p, int src, int dst,
                                                 uint8_t* vis, int* queue) {
    if (src == dst) {
        return false;
    }
    for (int i = 0; i < p; ++i) {
        vis[i] = 0;
    }
    int head = 0;
    int tail = 0;
    vis[src] = 1;
    queue[tail++] = src;
    while (head < tail) {
        const int u = queue[head++];
        const long long row = static_cast<long long>(u) * p;
        for (int v = 0; v < p; ++v) {
            if (vis[v] || G[row + v] != kArrow) {
                continue;
            }
            if (at(G, p, v, u) != kTail) {
                continue;
            }
            if (v == dst) {
                return true;
            }
            vis[v] = 1;
            queue[tail++] = v;
        }
    }
    return false;
}

__device__ __forceinline__ void orient(int8_t* G, int p, int src, int tgt, int* nchg,
                                       uint8_t* anc_vis, int* anc_queue) {
    if (is_dir(G, p, src, tgt) || is_bi(G, p, src, tgt)) {
        return;
    }
    if (is_dir(G, p, tgt, src)) {
        put(G, p, src, tgt, static_cast<int8_t>(kArrow));
        put(G, p, tgt, src, static_cast<int8_t>(kArrow));
        atomicAdd(nchg, 1);
        return;
    }
    if (is_und(G, p, src, tgt)) {
        // Cycle guard: skip when tgt is already an ancestor of src over
        // fully directed edges (sequential _orient_edge_conflict_aware).
        if (reaches_directed(G, p, tgt, src, anc_vis, anc_queue)) {
            return;
        }
        put(G, p, src, tgt, static_cast<int8_t>(kArrow));
        put(G, p, tgt, src, static_cast<int8_t>(kTail));
        atomicAdd(nchg, 1);
    }
}

__device__ __forceinline__ void comb_unrank(int t, int d, int& a, int& b) {
    int left = t;
    for (int a0 = 0; a0 < d - 1; ++a0) {
        const int nrest = d - 1 - a0;
        if (left < nrest) {
            a = a0;
            b = a0 + 1 + left;
            return;
        }
        left -= nrest;
    }
    a = d - 2;
    b = d - 1;
}

// First matching itertools.perm of {i,j,k}: a→b→c with a—c. Writes (a,c).
__device__ __forceinline__ bool r2_hit_pair(const int8_t* G, int p, int i, int j, int k, int& a,
                                            int& c) {
    const int v[3] = {i, j, k};
    const int pr[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (int r = 0; r < 6; ++r) {
        const int aa = v[pr[r][0]];
        const int b = v[pr[r][1]];
        const int cc = v[pr[r][2]];
        if (is_dir(G, p, aa, b) && is_dir(G, p, b, cc) && is_und(G, p, aa, cc)) {
            a = aa;
            c = cc;
            return true;
        }
    }
    return false;
}

// Unshielded V-shapes (i, j, k) at center j: i and k are neighbors of j,
// not adjacent, i before k in compacted neighbor order (vertex id).
// CSR by j so apply can walk one center at a time.
__global__ void pack_v_count_kernel(const int8_t* G, int p, const int* deg, const int* nbrs,
                                    int* counts) {
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= p) {
        return;
    }
    const int d = deg[j];
    const long long oj = static_cast<long long>(j) * p;
    int n = 0;
    for (int a = 0; a < d; ++a) {
        const int i = nbrs[oj + a];
        for (int b = a + 1; b < d; ++b) {
            if (!is_adj(G, p, i, nbrs[oj + b])) {
                ++n;
            }
        }
    }
    counts[j] = n;
}

__global__ void pack_v_fill_kernel(const int8_t* G, int p, const int* deg, const int* nbrs,
                                   int* v_i, int* v_k, const int* v_off) {
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= p) {
        return;
    }
    const int d = deg[j];
    const long long oj = static_cast<long long>(j) * p;
    int n = v_off[j];
    for (int a = 0; a < d; ++a) {
        const int i = nbrs[oj + a];
        for (int b = a + 1; b < d; ++b) {
            const int k = nbrs[oj + b];
            if (is_adj(G, p, i, k)) {
                continue;
            }
            v_i[n] = i;
            v_k[n] = k;
            ++n;
        }
    }
}

// 3-cliques i<j<k, CSR by smallest vertex i. Adjacency only.
__device__ __forceinline__ int r2_nbrs_after(const int* nbrs, long long oi, int d, int i) {
    int j0 = 0;
    while (j0 < d && nbrs[oi + j0] <= i) {
        ++j0;
    }
    return j0;
}

__global__ void pack_triangles_count_kernel(const int8_t* G, int p, const int* deg, const int* nbrs,
                                            int* counts) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= p) {
        return;
    }
    const int d = deg[i];
    const long long oi = static_cast<long long>(i) * p;
    const int j0 = r2_nbrs_after(nbrs, oi, d, i);
    int n = 0;
    for (int j_ind = j0; j_ind < d; ++j_ind) {
        const int j = nbrs[oi + j_ind];
        for (int k_ind = j_ind + 1; k_ind < d; ++k_ind) {
            if (is_adj(G, p, j, nbrs[oi + k_ind])) {
                ++n;
            }
        }
    }
    counts[i] = n;
}

__global__ void pack_triangles_fill_kernel(const int8_t* G, int p, const int* deg, const int* nbrs,
                                           int* t_i, int* t_j, int* t_k, const int* t_off) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= p) {
        return;
    }
    const int d = deg[i];
    const long long oi = static_cast<long long>(i) * p;
    const int j0 = r2_nbrs_after(nbrs, oi, d, i);
    int n = t_off[i];
    for (int j_ind = j0; j_ind < d; ++j_ind) {
        const int j = nbrs[oi + j_ind];
        for (int k_ind = j_ind + 1; k_ind < d; ++k_ind) {
            const int k = nbrs[oi + k_ind];
            if (!is_adj(G, p, j, k)) {
                continue;
            }
            t_i[n] = i;
            t_j[n] = j;
            t_k[n] = k;
            ++n;
        }
    }
}

// find_kites: outer l, combinadic neighbor pairs (j,k) of l with j not
// adjacent to k, then i in N(j)∩N(k)∩N(l). Python walks that
// intersection in N(j) insertion order (np.where → increasing id);
// walking compacted N(l) and keeping adj(i,j)&&adj(i,k) is the same
// sorted order. CSR k_off is pack-only; apply uses the flat kt_* list.
__global__ void pack_kites_count_kernel(const int8_t* G, int p, const int* deg, const int* nbrs,
                                        int* counts) {
    const int l = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (l >= p) {
        return;
    }
    const int d = deg[l];
    const long long ol = static_cast<long long>(l) * p;
    int n = 0;
    for (int a = 0; a < d; ++a) {
        const int j = nbrs[ol + a];
        for (int b = a + 1; b < d; ++b) {
            const int k = nbrs[ol + b];
            if (is_adj(G, p, j, k)) {
                continue;
            }
            for (int c = 0; c < d; ++c) {
                const int i = nbrs[ol + c];
                if (i == j || i == k) {
                    continue;
                }
                if (is_adj(G, p, i, j) && is_adj(G, p, i, k)) {
                    ++n;
                }
            }
        }
    }
    counts[l] = n;
}

__global__ void pack_kites_fill_kernel(const int8_t* G, int p, const int* deg, const int* nbrs,
                                       int* kt_i, int* kt_j, int* kt_k, int* kt_l,
                                       const int* k_off) {
    const int l = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (l >= p) {
        return;
    }
    const int d = deg[l];
    const long long ol = static_cast<long long>(l) * p;
    int n = k_off[l];
    for (int a = 0; a < d; ++a) {
        const int j = nbrs[ol + a];
        for (int b = a + 1; b < d; ++b) {
            const int k = nbrs[ol + b];
            if (is_adj(G, p, j, k)) {
                continue;
            }
            for (int c = 0; c < d; ++c) {
                const int i = nbrs[ol + c];
                if (i == j || i == k) {
                    continue;
                }
                if (!is_adj(G, p, i, j) || !is_adj(G, p, i, k)) {
                    continue;
                }
                kt_i[n] = i;
                kt_j[n] = j;
                kt_k[n] = k;
                kt_l[n] = l;
                ++n;
            }
        }
    }
}

__device__ __forceinline__ void cta_exscan(int want, int tid, int bdx, int* sh_off, int* sh_cnt) {
    sh_off[tid] = want;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int u = 0; u < bdx; ++u) {
            const int v = sh_off[u];
            sh_off[u] = acc;
            acc += v;
        }
        *sh_cnt = acc;
    }
    __syncthreads();
}

// CPU R3: one kite (i,j,k,l) is ready iff j→l, k→l, i—l, i—j, i—k.
__device__ __forceinline__ bool r3_hit_kite(const int8_t* G, int p, int i, int j, int k, int l) {
    return is_dir(G, p, j, l) && is_dir(G, p, k, l) && is_und(G, p, i, l) && is_und(G, p, i, j) &&
           is_und(G, p, i, k);
}

__global__ void compact_neighbors_kernel(const int8_t* G, int p, int* deg, int* nbrs,
                                         int* max_deg) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= p) {
        return;
    }
    int d = 0;
    const long long row = static_cast<long long>(i) * p;
    for (int j = 0; j < p; ++j) {
        if (G[row + j] != 0) {
            nbrs[row + d] = j;
            ++d;
        }
    }
    deg[i] = d;
    if (d > 0) {
        atomicMax(max_deg, d);
    }
}

__global__ void meek_kernel(int8_t* G, int p, int* stats, int rules, int* v_i, int* v_k, int* v_off,
                            int* t_i, int* t_j, int* t_k, int* t_off, const int* kt_i,
                            const int* kt_j, const int* kt_k, const int* kt_l, int n_kites,
                            uint8_t* anc_vis, int* anc_queue) {
    // One CTA, 256 threads. Integer endpoint matrix. Gauss–Seidel R1–R3.
    const int tid = static_cast<int>(threadIdx.x);
    const int bdx = static_cast<int>(blockDim.x);
    // Per-thread cycle-guard scratch rows (ancestor BFS).
    uint8_t* my_vis = anc_vis != nullptr ? anc_vis + static_cast<long long>(tid) * p : nullptr;
    int* my_queue = anc_queue != nullptr ? anc_queue + static_cast<long long>(tid) * p : nullptr;
    __shared__ int sh_off[kBlock];
    __shared__ int sh_n;
    __shared__ int sh_npp;
    __shared__ int sh_src;
    __shared__ int sh_next;

    int n_iters = 0;
    int n_orients = 0;
    const int limit = p * p > 0 ? p * p : 1;

    while (n_iters < limit) {
        if (tid == 0) {
            sh_n = 0;
        }
        __syncthreads();
        ++n_iters;

        // R1: frozen unshielded (i,k) at center j, CSR v_off[j]. Parallel
        // inside j (paints of one V do not spoil others at the same j),
        // then sync so later centers see the new arrows.
        if (rules & kRuleR1) {
            for (int j = 0; j < p; ++j) {
                const int b0 = v_off[j];
                const int n_v = v_off[j + 1] - b0;
                if (n_v == 0) {
                    continue;
                }
                for (int t = tid; t < n_v; t += bdx) {
                    const int i = v_i[b0 + t];
                    const int k = v_k[b0 + t];
                    if (is_dir(G, p, i, j) && is_und(G, p, j, k)) {
                        orient(G, p, j, k, &sh_n, my_vis, my_queue);
                    } else if (is_dir(G, p, k, j) && is_und(G, p, j, i)) {
                        orient(G, p, j, i, &sh_n, my_vis, my_queue);
                    }
                }
                __syncthreads();
            }
        }

        // R2: 3-cliques i<j<k packed once. 256-slot window keyed by packed
        // index g. The thread with g == min hit paints from registers.
        if (rules & kRuleR2) {
            const int ntri = t_off[p];
            int g = -1;
            int ti = 0;
            int tj = 0;
            int tk = 0;
            int a = 0;
            int c = 0;
            if (tid == 0) {
                sh_src = 0;
            }
            __syncthreads();
            while (true) {
                const int want = (g < 0) ? 1 : 0;
                cta_exscan(want, tid, bdx, sh_off, &sh_npp);
                const int src = sh_src;
                const int remain = ntri - src;
                const int nissue = sh_npp < remain ? sh_npp : remain;
                if (want && sh_off[tid] < nissue) {
                    g = src + sh_off[tid];
                    ti = t_i[g];
                    tj = t_j[g];
                    tk = t_k[g];
                }
                if (tid == 0) {
                    sh_src = src + nissue;
                }
                if (__syncthreads_count(g >= 0) == 0) {
                    break;
                }
                if (tid == 0) {
                    sh_next = 0x7fffffff;
                }
                __syncthreads();
                if (g >= 0 && r2_hit_pair(G, p, ti, tj, tk, a, c)) {
                    atomicMin(&sh_next, g);
                }
                __syncthreads();
                if (sh_next == 0x7fffffff) {
                    g = -1;
                    continue;
                }
                if (g == sh_next) {
                    orient(G, p, a, c, &sh_n, my_vis, my_queue);
                }
                __syncthreads();
                if (g >= 0 && g <= sh_next) {
                    g = -1;
                }
            }
        }

        // R3: packed kites (i,j,k,l) in find_kites order. Same 256-slot
        // batch-index window as R2: the thread with g == min hit paints
        // i→l from registers (ki, kl). g <= hit refill; suffix re-tests
        // (a new parent of l can enable a later kite). CPU does not
        // revisit earlier kites this pass.
        if (rules & kRuleR3) {
            int g = -1;
            int ki = 0;
            int kj = 0;
            int kk = 0;
            int kl = 0;
            if (tid == 0) {
                sh_src = 0;
            }
            __syncthreads();
            while (true) {
                const int want = (g < 0) ? 1 : 0;
                cta_exscan(want, tid, bdx, sh_off, &sh_npp);
                const int src = sh_src;
                const int remain = n_kites - src;
                const int nissue = sh_npp < remain ? sh_npp : remain;
                if (want && sh_off[tid] < nissue) {
                    g = src + sh_off[tid];
                    ki = kt_i[g];
                    kj = kt_j[g];
                    kk = kt_k[g];
                    kl = kt_l[g];
                }
                if (tid == 0) {
                    sh_src = src + nissue;
                }
                if (__syncthreads_count(g >= 0) == 0) {
                    break;
                }
                if (tid == 0) {
                    sh_next = 0x7fffffff;
                }
                __syncthreads();
                if (g >= 0) {
                    if (r3_hit_kite(G, p, ki, kj, kk, kl)) {
                        atomicMin(&sh_next, g);
                    }
                }
                __syncthreads();
                if (sh_next == 0x7fffffff) {
                    g = -1;
                    continue;
                }
                if (g == sh_next) {
                    orient(G, p, ki, kl, &sh_n, my_vis, my_queue);
                }
                __syncthreads();
                if (g >= 0 && g <= sh_next) {
                    g = -1;
                }
            }
        }

        n_orients += sh_n;
        __syncthreads();
        if (sh_n == 0) {
            break;
        }
        __syncthreads();
    }

    if (tid == 0) {
        stats[0] = n_iters;
        stats[1] = n_orients;
    }
}

__device__ __forceinline__ bool in_sep(const int64_t* S, const int64_t* ranges, int p, int x, int z,
                                       int y) {
    const long long base = (static_cast<long long>(x) * p + z) * 2;
    const int64_t st = ranges[base];
    const int64_t en = ranges[base + 1];
    if (st < 0 || en <= st) {
        return false;
    }
    for (int64_t i = st; i < en; ++i) {
        if (S[i] == y) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool sepset_recorded(const int64_t* ranges, int p, int x, int z) {
    const long long base = (static_cast<long long>(x) * p + z) * 2;
    return ranges[base] >= 0;
}

// Unshielded colliders, same order as CausalGraph.find_unshielded_triples:
// center j, then combinadic neighbor pairs. Per j: the CTA scans candidate
// triples (unshielded + sepset omits j) into a packed list, then one thread
// applies uc_priority=2 so later centers see earlier arrows.
__global__ void orient_colliders_kernel(int8_t* G, int p, const int64_t* S, const int64_t* ranges,
                                        int* deg, int* nbrs, int* cand_i, int* cand_k, int* reach,
                                        int* queue) {
    const int tid = static_cast<int>(threadIdx.x);
    const int bdx = static_cast<int>(blockDim.x);
    __shared__ int sh_off[kBlock];
    __shared__ int sh_n;
    __shared__ int sh_head;
    __shared__ int sh_tail;
    __shared__ int sh_next;

    for (int i = tid; i < p; i += bdx) {
        int d = 0;
        const long long row = static_cast<long long>(i) * p;
        for (int j = 0; j < p; ++j) {
            if (G[row + j] != 0) {
                nbrs[row + d] = j;
                ++d;
            }
        }
        deg[i] = d;
    }
    __syncthreads();

    for (int j = 0; j < p; ++j) {
        const int d = deg[j];
        if (d < 2) {
            continue;
        }
        // Orientations at center j only add arrows INTO j; j's outgoing
        // directed reach is therefore fixed for the whole center. Compute it
        // once with a CTA-wide frontier BFS instead of one BFS per candidate.
        for (int i = tid; i < p; i += bdx) {
            reach[i] = 0;
        }
        __syncthreads();
        if (tid == 0) {
            queue[0] = j;
            reach[j] = 1;
            sh_head = 0;
            sh_tail = 1;
        }
        __syncthreads();
        while (sh_head < sh_tail) {
            const int head = sh_head;
            const int tail = sh_tail;
            if (tid == 0) {
                sh_next = 0;
            }
            __syncthreads();
            for (int qi = head + tid; qi < tail; qi += bdx) {
                const int a = queue[qi];
                const int da = deg[a];
                const int* row = nbrs + static_cast<long long>(a) * p;
                for (int q = 0; q < da; ++q) {
                    const int b = row[q];
                    if (is_dir(G, p, a, b) && atomicCAS(&reach[b], 0, 1) == 0) {
                        const int pos = tail + atomicAdd(&sh_next, 1);
                        queue[pos] = b;
                    }
                }
            }
            __syncthreads();
            if (tid == 0) {
                sh_head = tail;
                sh_tail = tail + sh_next;
            }
            __syncthreads();
        }

        const long long off = static_cast<long long>(j) * p;
        const int npairs = d * (d - 1) / 2;
        int n_cand = 0;
        for (int base = 0; base < npairs; base += bdx) {
            const int t = base + tid;
            int i = 0;
            int k = 0;
            int want = 0;
            if (t < npairs) {
                int a, b;
                comb_unrank(t, d, a, b);
                i = nbrs[off + a];
                k = nbrs[off + b];
                if (!is_adj(G, p, i, k) &&
                    (sepset_recorded(ranges, p, i, k) || sepset_recorded(ranges, p, k, i)) &&
                    !in_sep(S, ranges, p, i, k, j) && !in_sep(S, ranges, p, k, i, j)) {
                    want = 1;
                }
            }
            sh_off[tid] = want;
            __syncthreads();
            if (tid == 0) {
                int acc = 0;
                for (int u = 0; u < bdx; ++u) {
                    const int v = sh_off[u];
                    sh_off[u] = acc;
                    acc += v;
                }
                sh_n = acc;
            }
            __syncthreads();
            if (want) {
                const int slot = n_cand + sh_off[tid];
                cand_i[slot] = i;
                cand_k[slot] = k;
            }
            n_cand += sh_n;
            __syncthreads();
        }

        if (tid == 0) {
            for (int c = 0; c < n_cand; ++c) {
                const int i = cand_i[c];
                const int k = cand_k[c];
                if (is_dir(G, p, j, i) || is_dir(G, p, j, k)) {
                    continue;
                }
                // Match the CPU's atomic candidate check: reject a collider
                // that would close a directed cycle (j already reaches i or k).
                if (reach[i] || reach[k]) {
                    continue;
                }
                put(G, p, i, j, static_cast<int8_t>(kArrow));
                put(G, p, j, i, static_cast<int8_t>(kTail));
                put(G, p, k, j, static_cast<int8_t>(kArrow));
                put(G, p, j, k, static_cast<int8_t>(kTail));
            }
        }
        __syncthreads();
    }
}

} // namespace

namespace {

std::tuple<int64_t, int64_t> meek_launch(torch::Tensor G, int rules) {
    const int p = static_cast<int>(G.size(0));
    if (p == 0) {
        return {1, 0};
    }
    const bool want_v = (rules & kRuleR1) != 0;
    const bool want_t = (rules & kRuleR2) != 0;
    const bool need_r3 = (rules & kRuleR3) != 0;
    auto deg = torch::empty({p}, G.options().dtype(at::kInt));
    auto nbrs = torch::empty({p, p}, G.options().dtype(at::kInt));
    auto max_deg = torch::zeros({1}, G.options().dtype(at::kInt));
    auto stats = torch::zeros({2}, G.options().dtype(at::kInt));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int compact_grid = (p + kBlock - 1) / kBlock;
    compact_neighbors_kernel<<<compact_grid, kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), p, deg.data_ptr<int>(), nbrs.data_ptr<int>(),
        max_deg.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    const int dmax = max_deg.cpu().item<int>();
    int64_t wedge = 1;
    if (p >= 1 && dmax >= 2) {
        wedge = static_cast<int64_t>(p) * dmax * (dmax - 1) / 2;
    }
    TORCH_CHECK(wedge <= static_cast<int64_t>(0x7fffffff), "Meek shape cap overflows int");
    const int vcap = want_v ? static_cast<int>(wedge) : 1;
    const int tcap = want_t ? static_cast<int>(wedge) : 1;
    auto v_i = torch::empty({vcap}, G.options().dtype(at::kInt));
    auto v_k = torch::empty({vcap}, G.options().dtype(at::kInt));
    auto v_off = torch::empty({p + 1}, G.options().dtype(at::kInt));
    auto t_i = torch::empty({tcap}, G.options().dtype(at::kInt));
    auto t_j = torch::empty({tcap}, G.options().dtype(at::kInt));
    auto t_k = torch::empty({tcap}, G.options().dtype(at::kInt));
    auto t_off = torch::empty({p + 1}, G.options().dtype(at::kInt));
    if (want_v) {
        auto counts = torch::empty({p}, G.options().dtype(at::kInt));
        pack_v_count_kernel<<<compact_grid, kBlock, 0, stream>>>(
            G.data_ptr<int8_t>(), p, deg.data_ptr<int>(), nbrs.data_ptr<int>(),
            counts.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        auto cs = counts.to(at::kLong).cumsum(0);
        const int64_t nv = (p == 0) ? 0 : cs[p - 1].item<int64_t>();
        TORCH_CHECK(nv <= static_cast<int64_t>(vcap), "Meek V-shape pack overflow n=", nv,
                    " cap=", vcap);
        v_off.narrow(0, 0, 1).zero_();
        if (p > 0) {
            v_off.narrow(0, 1, p).copy_(cs.to(at::kInt));
        }
        pack_v_fill_kernel<<<compact_grid, kBlock, 0, stream>>>(
            G.data_ptr<int8_t>(), p, deg.data_ptr<int>(), nbrs.data_ptr<int>(), v_i.data_ptr<int>(),
            v_k.data_ptr<int>(), v_off.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (want_t) {
        auto counts = torch::empty({p}, G.options().dtype(at::kInt));
        pack_triangles_count_kernel<<<compact_grid, kBlock, 0, stream>>>(
            G.data_ptr<int8_t>(), p, deg.data_ptr<int>(), nbrs.data_ptr<int>(),
            counts.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        auto cs = counts.to(at::kLong).cumsum(0);
        const int64_t ntri = (p == 0) ? 0 : cs[p - 1].item<int64_t>();
        TORCH_CHECK(ntri <= static_cast<int64_t>(tcap), "Meek triangle pack overflow n=", ntri,
                    " cap=", tcap);
        t_off.narrow(0, 0, 1).zero_();
        if (p > 0) {
            t_off.narrow(0, 1, p).copy_(cs.to(at::kInt));
        }
        pack_triangles_fill_kernel<<<compact_grid, kBlock, 0, stream>>>(
            G.data_ptr<int8_t>(), p, deg.data_ptr<int>(), nbrs.data_ptr<int>(), t_i.data_ptr<int>(),
            t_j.data_ptr<int>(), t_k.data_ptr<int>(), t_off.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto kt_i = torch::empty({1}, G.options().dtype(at::kInt));
    auto kt_j = torch::empty({1}, G.options().dtype(at::kInt));
    auto kt_k = torch::empty({1}, G.options().dtype(at::kInt));
    auto kt_l = torch::empty({1}, G.options().dtype(at::kInt));
    int n_kites = 0;
    if (need_r3) {
        auto counts = torch::empty({p}, G.options().dtype(at::kInt));
        pack_kites_count_kernel<<<compact_grid, kBlock, 0, stream>>>(
            G.data_ptr<int8_t>(), p, deg.data_ptr<int>(), nbrs.data_ptr<int>(),
            counts.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        auto cs = counts.to(at::kLong).cumsum(0);
        const int64_t nk = (p == 0) ? 0 : cs[p - 1].item<int64_t>();
        TORCH_CHECK(nk <= static_cast<int64_t>(0x7fffffff), "Meek kite pack overflows int n=", nk);
        n_kites = static_cast<int>(nk);
        const int kcap = n_kites > 0 ? n_kites : 1;
        kt_i = torch::empty({kcap}, G.options().dtype(at::kInt));
        kt_j = torch::empty({kcap}, G.options().dtype(at::kInt));
        kt_k = torch::empty({kcap}, G.options().dtype(at::kInt));
        kt_l = torch::empty({kcap}, G.options().dtype(at::kInt));
        auto k_off = torch::empty({p + 1}, G.options().dtype(at::kInt));
        k_off.narrow(0, 0, 1).zero_();
        if (p > 0) {
            k_off.narrow(0, 1, p).copy_(cs.to(at::kInt));
        }
        pack_kites_fill_kernel<<<compact_grid, kBlock, 0, stream>>>(
            G.data_ptr<int8_t>(), p, deg.data_ptr<int>(), nbrs.data_ptr<int>(),
            kt_i.data_ptr<int>(), kt_j.data_ptr<int>(), kt_k.data_ptr<int>(), kt_l.data_ptr<int>(),
            k_off.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    // Per-thread scratch for the Meek cycle-guard ancestor BFS: one visited
    // row and one queue row per thread of the single meek CTA.
    auto anc_vis = torch::zeros({static_cast<int64_t>(kBlock), static_cast<int64_t>(p)},
                                G.options().dtype(at::kByte));
    auto anc_queue = torch::empty({static_cast<int64_t>(kBlock), static_cast<int64_t>(p)},
                                  G.options().dtype(at::kInt));
    meek_kernel<<<1, kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), p, stats.data_ptr<int>(), rules, v_i.data_ptr<int>(),
        v_k.data_ptr<int>(), v_off.data_ptr<int>(), t_i.data_ptr<int>(), t_j.data_ptr<int>(),
        t_k.data_ptr<int>(), t_off.data_ptr<int>(), kt_i.data_ptr<int>(), kt_j.data_ptr<int>(),
        kt_k.data_ptr<int>(), kt_l.data_ptr<int>(), n_kites, anc_vis.data_ptr<uint8_t>(),
        anc_queue.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    auto h = stats.cpu();
    return {static_cast<int64_t>(h[0].item<int>()), static_cast<int64_t>(h[1].item<int>())};
}

std::tuple<int64_t, int64_t> meek_orient_cuda(torch::Tensor G, int64_t rules) {
    TORCH_CHECK(G.is_cuda(), "meek G must be CUDA");
    TORCH_CHECK(G.dim() == 2 && G.size(0) == G.size(1), "meek G must be square");
    TORCH_CHECK(G.scalar_type() == at::kChar, "meek G must be int8");
    TORCH_CHECK(G.is_contiguous(), "meek G must be contiguous");
    TORCH_CHECK(rules >= 1 && rules <= 7, "rules bitmask 1=R1 2=R2 4=R3");
    return meek_launch(G, static_cast<int>(rules));
}

} // namespace

void meek_orient_colliders_cuda(torch::Tensor G, torch::Tensor S, torch::Tensor ranges) {
    TORCH_CHECK(G.is_cuda(), "collider G must be CUDA");
    TORCH_CHECK(G.dim() == 2 && G.size(0) == G.size(1), "collider G must be square");
    TORCH_CHECK(G.scalar_type() == at::kChar, "collider G must be int8");
    TORCH_CHECK(G.is_contiguous(), "collider G must be contiguous");
    const int p = static_cast<int>(G.size(0));
    TORCH_CHECK(!(G.diagonal() != 0).any().item<bool>(),
                "collider G must have a zero diagonal (self-loop in adjacency)");
    require_sepset_csr_cuda(S, ranges, p, "meek_orient_colliders");
    if (p == 0) {
        return;
    }
    auto deg = torch::empty({p}, G.options().dtype(at::kInt));
    auto nbrs = torch::empty({p, p}, G.options().dtype(at::kInt));
    // deg <= p-1 for a valid (zero-diagonal) adjacency. p*(p-1)/2 is still
    // safe if a self-loop ever reaches the kernel.
    const int64_t max_pairs =
        (p < 2) ? 1 : static_cast<int64_t>(p) * static_cast<int64_t>(p - 1) / 2;
    auto cand_i = torch::empty({max_pairs}, G.options().dtype(at::kInt));
    auto cand_k = torch::empty({max_pairs}, G.options().dtype(at::kInt));
    auto reach = torch::empty({p}, G.options().dtype(at::kInt));
    auto queue = torch::empty({p}, G.options().dtype(at::kInt));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    orient_colliders_kernel<<<1, kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), p, S.data_ptr<int64_t>(), ranges.data_ptr<int64_t>(),
        deg.data_ptr<int>(), nbrs.data_ptr<int>(), cand_i.data_ptr<int>(), cand_k.data_ptr<int>(),
        reach.data_ptr<int>(), queue.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void meek_preload() {
    cuda_preload_context();
    touch_cuda_kernel(compact_neighbors_kernel);
    touch_cuda_kernel(pack_v_count_kernel);
    touch_cuda_kernel(pack_v_fill_kernel);
    touch_cuda_kernel(pack_triangles_count_kernel);
    touch_cuda_kernel(pack_triangles_fill_kernel);
    touch_cuda_kernel(pack_kites_count_kernel);
    touch_cuda_kernel(pack_kites_fill_kernel);
    touch_cuda_kernel(meek_kernel);
    touch_cuda_kernel(orient_colliders_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("orient", &meek_orient_cuda, pybind11::arg("G"), pybind11::arg("rules") = 7);
    m.def("orient_colliders", &meek_orient_colliders_cuda);
    m.def("preload", &meek_preload);
}
