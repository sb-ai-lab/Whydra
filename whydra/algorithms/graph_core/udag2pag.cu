// FCI orientation on a dense endpoint matrix (int8).
//
// Marks: 0 none, 1 circle, 2 arrow, 3 tail.
// R1 packs freeze (a,b) with pag[a,b]==2 and reverse nonzero, in (b, a)
// order, then one CTA grid-strides c per pair (paints of one c do not spoil
// other c for that pair). R2 packs freeze pairs (a,c) in (c, a) order and
// uses a 256-slot window (live witness). R3 packs freeze pairs (b,d) in
// (d, b) order. R4/R6 reuse the R3 pair list (circle at b); R5 packs o–o
// in (b, a) order. R7 reuses the R3 pair list; R8/R9/R10 pack a *→ c with
// circle at a, in (c, a) order. R4–R8 and R10 sequential-apply (a hit can
// change later hits). R9 writes do not change other pairs' PD-paths; apply
// uses a block-cooperative search for an uncovered path.

#include <chrono>
#include <tuple>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_check.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_touch.cuh"
#include "cuda_validate.cuh"

namespace {

constexpr int kBlock = 256;

void prefix_offsets(const torch::Tensor& counts, torch::Tensor off) {
    const int p = static_cast<int>(counts.size(0));
    off.narrow(0, 0, 1).zero_();
    if (p > 0) {
        off.narrow(0, 1, p).copy_(counts.to(at::kLong).cumsum(0).to(at::kInt));
    }
}

__device__ __forceinline__ int8_t at(const int8_t* G, int p, int i, int j) {
    return G[static_cast<long long>(i) * p + j];
}

__device__ __forceinline__ void put(int8_t* G, int p, int i, int j, int8_t v) {
    G[static_cast<long long>(i) * p + j] = v;
}

__device__ __forceinline__ bool pd_edge(const int8_t* G, int p, int u, int v) {
    const int8_t uv = at(G, p, u, v);
    const int8_t vu = at(G, p, v, u);
    return (uv == 1 || uv == 2) && (vu == 1 || vu == 3);
}

__device__ __forceinline__ bool is_unf(const int32_t* unf, int U, int x, int y, int z) {
    for (int i = 0; i < U; ++i) {
        const int a = unf[i * 3 + 0];
        const int b = unf[i * 3 + 1];
        const int c = unf[i * 3 + 2];
        if ((a == x && b == y && c == z) || (a == z && b == y && c == x)) {
            return true;
        }
    }
    return false;
}

// Zhang (2008), R1--R10. Matrix entries name the mark at the column vertex.
struct Marks {
    const int8_t* matrix;
    int n;
    __device__ int operator()(int from, int to) const { return at(matrix, n, from, to); }
    __device__ bool apart(int u, int v) const { return !(*this)(u, v) && !(*this)(v, u); }
    __device__ bool circles(int u, int v) const { return (*this)(u, v) == 1 && (*this)(v, u) == 1; }
};

__device__ bool admissible_cycle(const int* path, int length, const int32_t* unf, int U) {
    if (U == 0)
        return true;
    // Center each excluded triple in turn, including at the join of a cycle.
    for (int center = 0; center < length; ++center) {
        const int prev = center ? center - 1 : length - 1;
        const int next = center + 1 == length ? 0 : center + 1;
        if (is_unf(unf, U, path[prev], path[center], path[next]))
            return false;
    }
    return true;
}

__device__ void sort_ints(int* a, int n) {
    for (int i = 1; i < n; ++i) {
        const int v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = v;
    }
}

enum class Neighbors {
    open_first_hop,
    open_second_hop,
    possible_child,
    incoming_head,
    circle_edge,
    r4_collider,
    exposed_circle,
    parent,
    first_hop
};

__device__ int compact_nodes(const int8_t* pag, const uint8_t* visited, int p, int* lst,
                             Neighbors selection, int from, int extra, int skip) {
    const int tid = static_cast<int>(threadIdx.x);
    const int bdx = static_cast<int>(blockDim.x);
    __shared__ int sh_n;
    if (tid == 0) {
        sh_n = 0;
    }
    __syncthreads();
    for (int i = tid; i < p; i += bdx) {
        bool ok = false;
        if (selection == Neighbors::open_first_hop) {
            ok = i != skip && pd_edge(pag, p, from, i) && at(pag, p, extra, i) == 0 &&
                 at(pag, p, i, extra) == 0;
        } else if (selection == Neighbors::open_second_hop) {
            ok = visited[i] == 0 && pd_edge(pag, p, from, i) && at(pag, p, i, extra) == 0;
        } else if (selection == Neighbors::possible_child) {
            ok = visited[i] == 0 && pd_edge(pag, p, from, i);
        } else if (selection == Neighbors::incoming_head) {
            ok = visited[i] == 0 && at(pag, p, from, i) != 0 && at(pag, p, i, from) == 2;
        } else if (selection == Neighbors::circle_edge) {
            ok = visited[i] == 0 && at(pag, p, from, i) == 1 && at(pag, p, i, from) == 1;
        } else if (selection == Neighbors::r4_collider) {
            ok = at(pag, p, from, i) == 2 && at(pag, p, i, from) != 0 &&
                 at(pag, p, extra, i) == 3 && at(pag, p, i, extra) == 2;
        } else if (selection == Neighbors::exposed_circle) {
            ok = i != skip && at(pag, p, from, i) == 1 && at(pag, p, i, from) == 1 &&
                 at(pag, p, extra, i) == 0 && at(pag, p, i, extra) == 0;
        } else if (selection == Neighbors::parent) {
            ok = at(pag, p, from, i) == 3 && at(pag, p, i, from) == 2;
        } else if (selection == Neighbors::first_hop) {
            ok = i != skip && pd_edge(pag, p, from, i);
        }
        if (ok) {
            const int s = atomicAdd(&sh_n, 1);
            lst[s] = i;
        }
    }
    __syncthreads();
    if (tid == 0) {
        sort_ints(lst, sh_n);
    }
    __syncthreads();
    const int count = sh_n;
    __syncthreads();
    return count;
}

__device__ __forceinline__ bool in_sep(const int64_t* S, const int64_t* ranges, int p, int x, int z,
                                       int y) {
    if (ranges == nullptr) {
        return false;
    }
    const long long base = (static_cast<long long>(x) * p + z) * 2;
    const int64_t st = ranges[base];
    const int64_t en = ranges[base + 1];
    if (st < 0 || en <= st || S == nullptr) {
        return false;
    }
    for (int64_t i = st; i < en; ++i) {
        if (S[i] == y) {
            return true;
        }
    }
    return false;
}

// R1 freeze: a *→ b, i.e. pag[a,b]==2 and reverse nonzero. CPU sorts (b, a).
__device__ __forceinline__ bool r1_pair(const int8_t* pag, int p, int a, int b) {
    return at(pag, p, a, b) == 2 && at(pag, p, b, a) != 0;
}

__global__ void pack_r1_count_kernel(const int8_t* pag, int p, int* counts) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= p) {
        return;
    }
    int n = 0;
    for (int a = 0; a < p; ++a) {
        if (r1_pair(pag, p, a, b)) {
            ++n;
        }
    }
    counts[b] = n;
}

__global__ void pack_r1_fill_kernel(const int8_t* pag, int p, const int* off, int* r1_a,
                                    int* r1_b) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= p) {
        return;
    }
    int n = off[b];
    for (int a = 0; a < p; ++a) {
        if (!r1_pair(pag, p, a, b)) {
            continue;
        }
        r1_a[n] = a;
        r1_b[n] = b;
        ++n;
    }
}

// Packed R1 apply. Sequential packed index g (CPU (b, a) order); grid-stride c.
__global__ void r1_kernel(int8_t* pag, int p, const int32_t* unf, int U, const int* r1_a,
                          const int* r1_b, int n_r1) {
    const Marks graph{pag, p};
    for (int pair = 0; pair != n_r1; ++pair) {
        const int center = r1_b[pair], source = r1_a[pair];
        for (int u = threadIdx.x; u < p; u += blockDim.x) {
            if (graph(u, center) != 1 || !graph(center, u))
                continue;
            if (source != u && graph.apart(source, u) && !is_unf(unf, U, source, center, u)) {
                put(pag, p, u, center, 3);
                put(pag, p, center, u, 2);
            }
        }
        __syncthreads();
    }
}

// R2 freeze: a o—* c, i.e. pag[a,c]==1 and reverse nonzero. CPU sorts (c, a).
__device__ __forceinline__ bool r2_pair(const int8_t* pag, int p, int a, int c) {
    return at(pag, p, a, c) == 1 && at(pag, p, c, a) != 0;
}

// Live witness b = i: a → i → c or a *→ i ←* c with tail at c (v2 pag[i][a]).
__device__ __forceinline__ bool r2_witness(const int8_t* pag, int p, int a, int c) {
    const Marks graph{pag, p};
    for (int middle = 0; middle != p; ++middle) {
        if (graph(a, middle) != 2 || graph(middle, c) != 2)
            continue;
        const int first = graph(middle, a), second = graph(c, middle);
        if (first && second && (first == 3 || second == 3))
            return true;
    }
    return false;
}

__global__ void pack_r2_count_kernel(const int8_t* pag, int p, int* counts) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= p) {
        return;
    }
    int n = 0;
    for (int a = 0; a < p; ++a) {
        if (r2_pair(pag, p, a, c)) {
            ++n;
        }
    }
    counts[c] = n;
}

__global__ void pack_r2_fill_kernel(const int8_t* pag, int p, const int* off, int* r2_a,
                                    int* r2_c) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= p) {
        return;
    }
    int n = off[c];
    for (int a = 0; a < p; ++a) {
        if (!r2_pair(pag, p, a, c)) {
            continue;
        }
        r2_a[n] = a;
        r2_c[n] = c;
        ++n;
    }
}

// R3 freeze: b *—* d with circle at b, i.e. pag[b,d]!=0 and pag[d,b]==1.
// CPU sorts (d, b).
__device__ __forceinline__ bool r3_pair(const int8_t* pag, int p, int b, int d) {
    return at(pag, p, b, d) != 0 && at(pag, p, d, b) == 1;
}

__device__ __forceinline__ bool r3_ac(const int8_t* pag, int p, int b, int d, int i) {
    return r1_pair(pag, p, i, b) && r2_pair(pag, p, i, d);
}

// Live: two unshielded members of indAC (sorted i < k). Empty unf orients
// any such pair; non-empty also skips unfaithful (i, b, k).
__device__ __forceinline__ bool r3_ready(const int8_t* pag, int p, int b, int d, const int32_t* unf,
                                         int U) {
    const Marks graph{pag, p};
    // Store eligible endpoints in word-sized groups. Pair membership is
    // computed once per group, then tested against earlier unshielded members.
    for (int base = 0; base < p; base += 32) {
        unsigned candidates = 0;
        for (int bit = 0; bit < 32 && base + bit < p; ++bit) {
            if (r3_ac(pag, p, b, d, base + bit))
                candidates |= 1u << bit;
        }
        if (!candidates)
            continue;
        for (int right = base + 1; right < p; ++right) {
            if (!r3_ac(pag, p, b, d, right))
                continue;
            for (unsigned bits = candidates; bits; bits &= bits - 1) {
                const int left = base + __ffs(bits) - 1;
                if (left >= right)
                    break;
                if (graph.apart(left, right) && !is_unf(unf, U, left, b, right))
                    return true;
            }
        }
    }
    return false;
}

__global__ void pack_r3_count_kernel(const int8_t* pag, int p, int* counts) {
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= p) {
        return;
    }
    int n = 0;
    for (int b = 0; b < p; ++b) {
        if (r3_pair(pag, p, b, d)) {
            ++n;
        }
    }
    counts[d] = n;
}

__global__ void pack_r3_fill_kernel(const int8_t* pag, int p, const int* off, int* r3_b,
                                    int* r3_d) {
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= p) {
        return;
    }
    int n = off[d];
    for (int b = 0; b < p; ++b) {
        if (!r3_pair(pag, p, b, d)) {
            continue;
        }
        r3_b[n] = b;
        r3_d[n] = d;
        ++n;
    }
}

// R5 freeze: a o–o b, i.e. pag[a,b]==1 and pag[b,a]==1. CPU sorts (b, a).
__device__ __forceinline__ bool r5_pair(const int8_t* pag, int p, int a, int b) {
    return at(pag, p, a, b) == 1 && at(pag, p, b, a) == 1;
}

__global__ void pack_r5_count_kernel(const int8_t* pag, int p, int* counts) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= p) {
        return;
    }
    int n = 0;
    for (int a = 0; a < p; ++a) {
        if (r5_pair(pag, p, a, b)) {
            ++n;
        }
    }
    counts[b] = n;
}

__global__ void pack_r5_fill_kernel(const int8_t* pag, int p, const int* off, int* r5_a,
                                    int* r5_b) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= p) {
        return;
    }
    int n = off[b];
    for (int a = 0; a < p; ++a) {
        if (!r5_pair(pag, p, a, b)) {
            continue;
        }
        r5_a[n] = a;
        r5_b[n] = b;
        ++n;
    }
}

// R8/R9/R10 freeze: a *→ c with circle at a, i.e. pag[a,c]==2 and pag[c,a]==1.
// CPU sorts (c, a).
__device__ __forceinline__ bool r8_pair(const int8_t* pag, int p, int a, int c) {
    return at(pag, p, a, c) == 2 && at(pag, p, c, a) == 1;
}

__global__ void pack_r8_count_kernel(const int8_t* pag, int p, int* counts) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= p) {
        return;
    }
    int n = 0;
    for (int a = 0; a < p; ++a) {
        if (r8_pair(pag, p, a, c)) {
            ++n;
        }
    }
    counts[c] = n;
}

__global__ void pack_r8_fill_kernel(const int8_t* pag, int p, const int* off, int* r8_a,
                                    int* r8_c) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= p) {
        return;
    }
    int n = off[c];
    for (int a = 0; a < p; ++a) {
        if (!r8_pair(pag, p, a, c)) {
            continue;
        }
        r8_a[n] = a;
        r8_c[n] = c;
        ++n;
    }
}

__device__ void set_leak(int* leak, int v) {
    if (leak != nullptr && threadIdx.x == 0) {
        *leak = v;
    }
}

// Device work buffers belong to one block. Entries hold a vertex and its
// predecessor slot. Expand equal-length witnesses in ascending neighbor order.
enum class Witness { discriminating, circle, possible };
struct PathScratch {
    uint8_t* expanded;
    int* ends;
    int* parents;
    int capacity;
    int* path;
    int* neighbors;
};

// For R4 use (a,b,c) = (first collider, discriminated vertex, last vertex).
// For R5/R9 use (a,b,c) = (first vertex, first hop, path target). R5 also
// closes the cycle at last, adjacent to c. Return the complete ordered path.
__device__ int find_witness(Witness kind, Marks graph, int a, int b, int c, int last,
                            const int32_t* unf, int U, PathScratch w) {
    const int lane = threadIdx.x;
    const bool disc = kind == Witness::discriminating, circle = kind == Witness::circle;
    const int root = disc ? a : b;
    for (int u = lane; u < graph.n; u += blockDim.x)
        w.expanded[u] = u == a || u == b || u == c || u == last;
    __syncthreads();
    int tail = 0;
    // Slot -1 expands the root, each following slot expands the vertex there.
    for (int slot = -1; slot < tail; ++slot) {
        const int v = slot < 0 ? root : w.ends[slot];
        if (slot >= 0) {
            if (lane == 0)
                w.expanded[v] = 1;
            __syncthreads();
            const bool terminal =
                disc ? graph.apart(v, c)
                     : (circle ? graph.circles(v, c) : pd_edge(graph.matrix, graph.n, v, c));
            if (terminal) {
                __shared__ int length;
                if (lane == 0) {
                    int n = 0;
                    for (int back = slot; back >= 0; back = w.parents[back])
                        w.path[n++] = w.ends[back];
                    w.path[n++] = root;
                    if (disc) {
                        w.path[n++] = b;
                        w.path[n++] = c;
                    } else {
                        w.path[n++] = a;
                        for (int i = 0; i < n / 2; ++i) {
                            const int opposite = n - i - 1, temp = w.path[i];
                            w.path[i] = w.path[opposite];
                            w.path[opposite] = temp;
                        }
                        w.path[n++] = c;
                        if (circle)
                            w.path[n++] = last;
                        bool uncovered = true;
                        for (int middle = 1; middle + 1 < n && uncovered; ++middle)
                            uncovered = graph.apart(w.path[middle - 1], w.path[middle + 1]);
                        if (!uncovered || !admissible_cycle(w.path, n, unf, U))
                            n = 0;
                    }
                    length = n;
                }
                __syncthreads();
                const int n = length;
                __syncthreads();
                if (n)
                    return n;
                continue;
            }
            if (disc) {
                const int prev = w.parents[slot] == -1 ? root : w.ends[w.parents[slot]];
                if (graph(v, c) != 2 || graph(c, v) != 3 || graph(prev, v) != 2)
                    continue;
            }
        }
        const Neighbors selection = disc       ? Neighbors::incoming_head
                                    : circle   ? Neighbors::circle_edge
                                    : slot < 0 ? Neighbors::open_second_hop
                                               : Neighbors::possible_child;
        const int count =
            compact_nodes(graph.matrix, w.expanded, graph.n, w.neighbors, selection, v, a, -1);
        if (count > w.capacity - tail)
            return 0;
        for (int index = lane; index < count; index += blockDim.x) {
            w.ends[tail + index] = w.neighbors[index];
            w.parents[tail + index] = slot;
        }
        tail += count;
        __syncthreads();
    }
    return 0;
}

// Evaluate a window of pairs in parallel. The first successful index is
// committed before rechecking later pairs against the updated marks.
template <bool diamond>
__device__ void apply_witness_windows(int8_t* pag, int p, const int* left, const int* right,
                                      int count, const int32_t* unf, int U) {
    __shared__ int first;
    for (int base = 0; base < count; base += blockDim.x) {
        const int index = base + threadIdx.x;
        int cursor = base;
        while (cursor < count && cursor - base < static_cast<int>(blockDim.x)) {
            if (!threadIdx.x)
                first = count;
            __syncthreads();
            if (index < count && index >= cursor) {
                const bool found = diamond ? r3_ready(pag, p, left[index], right[index], unf, U)
                                           : r2_witness(pag, p, left[index], right[index]);
                if (found)
                    atomicMin(&first, index);
            }
            __syncthreads();
            const int chosen = first;
            if (!threadIdx.x && chosen < count) {
                const int from = diamond ? right[chosen] : left[chosen];
                const int to = diamond ? left[chosen] : right[chosen];
                put(pag, p, from, to, 2);
            }
            __syncthreads();
            if (chosen == count)
                break;
            cursor = chosen + 1;
        }
    }
}

__global__ void r2_kernel(int8_t* pag, int p, const int* a, const int* c, int count) {
    apply_witness_windows<false>(pag, p, a, c, count, nullptr, 0);
}

__global__ void r3_kernel(int8_t* pag, int p, const int32_t* unf, int U, const int* b, const int* d,
                          int count) {
    apply_witness_windows<true>(pag, p, b, d, count, unf, U);
}

// Packed R4. Sequential g in CPU (c, b) order (same pairs as R3). A hit
// consumes the circle and can change later indA / minDiscPath, so apply
// is sequential.
__global__ void r4_apply_kernel(int8_t* pag, int p, const int64_t* S, const int64_t* ranges,
                                uint8_t* visited, int* q_node, int* q_pred, int qmax, int* path,
                                int* lst, int* indA, const int* r4_b, const int* r4_c, int n_r4,
                                int* leak) {
    const Marks graph{pag, p};
    const PathScratch work{visited, q_node, q_pred, qmax, path, lst};
    for (int edge = 0; edge != n_r4; ++edge) {
        const int center = r4_b[edge], target = r4_c[edge];
        set_leak(leak, target);
        const int count =
            compact_nodes(pag, nullptr, p, indA, Neighbors::r4_collider, center, target, -1);
        for (int i = 0; i < count && graph(target, center) == 1; ++i) {
            const int before = indA[i];
            const int length = find_witness(Witness::discriminating, graph, before, center, target,
                                            -1, nullptr, 0, work);
            if (!length)
                continue;
            if (!threadIdx.x) {
                const int first = path[0];
                const bool separated = in_sep(S, ranges, p, first, target, center) ||
                                       in_sep(S, ranges, p, target, first, center);
                put(pag, p, target, center, separated ? 3 : 2);
                put(pag, p, center, target, 2);
                if (!separated)
                    put(pag, p, before, center, 2);
            }
            __syncthreads();
        }
    }
}

// Packed R5. Sequential g in CPU (b, a) order. Tails disable later o–o /
// uncovered cycles, so apply is sequential.
__global__ void r5_apply_kernel(int8_t* pag, int p, const int32_t* unf, int U, uint8_t* visited,
                                int* q_node, int* q_pred, int qmax, int* path, int* lst, int* indC,
                                int* indD, const int* r5_a, const int* r5_b, int n_r5, int* leak) {
    const Marks graph{pag, p};
    const PathScratch work{visited, q_node, q_pred, qmax, path, lst};
    for (int edge = 0; edge != n_r5; ++edge) {
        const int a = r5_a[edge], b = r5_b[edge];
        const int left = compact_nodes(pag, nullptr, p, indC, Neighbors::exposed_circle, a, b, b);
        const int right = compact_nodes(pag, nullptr, p, indD, Neighbors::exposed_circle, b, a, a);
        if (!right)
            continue;
        for (int i = 0; i < left && graph(a, b) == 1; ++i) {
            const int first = indC[i];
            set_leak(leak, first);
            for (int j = 0; j < right; ++j) {
                const int end = indD[j];
                int length;
                if (graph.circles(first, end)) {
                    if (!threadIdx.x) {
                        path[0] = a;
                        path[1] = first;
                        path[2] = end;
                        path[3] = b;
                    }
                    __syncthreads();
                    length = admissible_cycle(path, 4, unf, U) ? 4 : 0;
                    __syncthreads();
                } else {
                    length = find_witness(Witness::circle, graph, a, first, end, b, unf, U, work);
                }
                if (!length)
                    continue;
                for (int k = threadIdx.x; k < length; k += blockDim.x) {
                    const int u = path[k], v = path[k + 1 == length ? 0 : k + 1];
                    put(pag, p, u, v, 3);
                    put(pag, p, v, u, 3);
                }
                __syncthreads();
                break;
            }
        }
    }
}

// Packed R6. Sequential g: a new undirected at b can enable a later pair
// with the same b. Parallel OR over i; sequential apply.
__global__ void r6_kernel(int8_t* pag, int p, const int* r6_b, const int* r6_c, int n_r6) {
    const int tid = static_cast<int>(threadIdx.x);
    const int bdx = static_cast<int>(blockDim.x);
    __shared__ int sh_hit;
    for (int g = 0; g < n_r6; ++g) {
        const int b = r6_b[g];
        const int c = r6_c[g];
        if (tid == 0) {
            sh_hit = 0;
        }
        __syncthreads();
        for (int i = tid; i < p; i += bdx) {
            if (at(pag, p, b, i) == 3 && at(pag, p, i, b) == 3) {
                sh_hit = 1;
            }
        }
        __syncthreads();
        if (sh_hit && tid == 0) {
            put(pag, p, c, b, 3);
        }
        __syncthreads();
    }
}

// Packed R7. Sequential g: writing pag[c,b]=3 can create a new R7 witness
// at c. Parallel OR over i; sequential apply.
__global__ void r7_kernel(int8_t* pag, int p, const int32_t* unf, int U, const int* r7_b,
                          const int* r7_c, int n_r7) {
    const Marks graph{pag, p};
    for (int edge = 0; edge != n_r7; ++edge) {
        const int center = r7_b[edge], end = r7_c[edge];
        bool possible = false;
        for (int u = threadIdx.x; u < p && !possible; u += blockDim.x)
            possible = u != end && graph(u, center) == 1 && graph(center, u) == 3 &&
                       graph.apart(u, end) && !is_unf(unf, U, u, center, end);
        if (__syncthreads_or(possible) && !threadIdx.x)
            put(pag, p, end, center, 3);
        __syncthreads();
    }
}

// Packed R8. Sequential g: a new tail at a can enable a later pair with
// the same a. Parallel OR over i.
__global__ void r8_kernel(int8_t* pag, int p, const int* r8_a, const int* r8_c, int n_r8) {
    const int tid = static_cast<int>(threadIdx.x);
    const int bdx = static_cast<int>(blockDim.x);
    __shared__ int sh_hit;
    for (int g = 0; g < n_r8; ++g) {
        const int a = r8_a[g];
        const int c = r8_c[g];
        if (tid == 0) {
            sh_hit = 0;
        }
        __syncthreads();
        for (int i = tid; i < p; i += bdx) {
            if (at(pag, p, i, a) == 3 && (at(pag, p, a, i) == 2 || at(pag, p, a, i) == 1) &&
                at(pag, p, c, i) == 3 && at(pag, p, i, c) == 2) {
                sh_hit = 1;
            }
        }
        __syncthreads();
        if (sh_hit && tid == 0) {
            put(pag, p, c, a, 3);
        }
        __syncthreads();
    }
}

// Packed R9. Sequential g in CPU (c, a) order. A hit paints pag[c,a]=3
// and does not change other pairs' PD-paths; remaining b's of this pair
// still stop.
__global__ void r9_apply_kernel(int8_t* pag, int p, const int32_t* unf, int U, uint8_t* visited,
                                int* q_node, int* q_pred, int qmax, int* path, int* lst, int* indB,
                                const int* r9_a, const int* r9_c, int n_r9, int* leak) {
    const Marks graph{pag, p};
    const PathScratch work{visited, q_node, q_pred, qmax, path, lst};
    for (int edge = 0; edge != n_r9; ++edge) {
        const int start = r9_a[edge], end = r9_c[edge];
        set_leak(leak, end);
        if (graph(end, start) != 1)
            continue;
        const int choices =
            compact_nodes(pag, nullptr, p, indB, Neighbors::open_first_hop, start, end, end);
        for (int i = 0; i < choices; ++i) {
            if (find_witness(Witness::possible, graph, start, indB[i], end, -1, unf, U, work)) {
                if (!threadIdx.x)
                    put(pag, p, end, start, 3);
                __syncthreads();
                break;
            }
        }
    }
}

// Zhang (2008), rule R10, doi:10.1016/j.artint.2008.08.001.
// Group witnesses by their first hop and join groups that can orient the edge.
constexpr int kParentWordBits = 32;

__device__ bool r10_direct_pair(Marks graph, int start, int left, int right) {
    return graph.apart(left, right) && pd_edge(graph.matrix, graph.n, start, left) &&
           pd_edge(graph.matrix, graph.n, start, right);
}

__device__ bool r10_has_direct_witness(Marks graph, int start, const int* parents, int count,
                                       const int32_t* unf, int U) {
    bool found = false;
    for (int i = threadIdx.x; i < count && !found; i += blockDim.x) {
        const int parent = parents[i];
        for (int j = 0; j < i; ++j) {
            const int other = parents[j];
            if (r10_direct_pair(graph, start, parent, other) &&
                !is_unf(unf, U, parent, start, other)) {
                found = true;
                break;
            }
        }
    }
    return __syncthreads_or(found);
}

__device__ bool r10_has_compatible_parent(Marks graph, int start, int hop, int parent,
                                          const int* parents, const int* witness_hops,
                                          int witness_count, const uint32_t* parents_by_hop,
                                          int words_per_hop, const int32_t* unf, int U) {
    bool found = false;
    for (int row = threadIdx.x; row < witness_count && !found; row += blockDim.x) {
        const int other_hop = witness_hops[row];
        if ((graph(hop, other_hop) && graph(other_hop, hop)) ||
            is_unf(unf, U, hop, start, other_hop)) {
            continue;
        }
        const uint32_t* reachable = parents_by_hop + static_cast<int64_t>(row) * words_per_hop;
        for (int word = 0; word < words_per_hop && !found; ++word) {
            for (uint32_t bits = reachable[word]; bits; bits &= bits - 1) {
                const int other = parents[word * kParentWordBits + __ffs(bits) - 1];
                // An ambiguous direct pair is excluded from the long-path search.
                if (other != parent && !r10_direct_pair(graph, start, parent, other)) {
                    found = true;
                    break;
                }
            }
        }
    }
    return __syncthreads_or(found);
}

__device__ bool r10_has_long_witness(Marks graph, int start, const int* parents, int parent_count,
                                     int* first_hops, int hop_count, uint32_t* parents_by_hop,
                                     const int32_t* unf, int U, PathScratch work) {
    const int words_per_hop = (parent_count - 1) / kParentWordBits + 1;
    int witness_count = 0;
    for (int h = 0; h < hop_count; ++h) {
        const int hop = first_hops[h];
        uint32_t* reachable = parents_by_hop + static_cast<int64_t>(witness_count) * words_per_hop;
        bool has_path = false;
        for (int i = 0; i < parent_count; ++i) {
            const int parent = parents[i];
            if (!find_witness(Witness::possible, graph, start, hop, parent, -1, unf, U, work)) {
                continue;
            }
            if (witness_count &&
                r10_has_compatible_parent(graph, start, hop, parent, parents, first_hops,
                                          witness_count, parents_by_hop, words_per_hop, unf, U)) {
                return true;
            }
            if (!has_path) {
                for (int word = threadIdx.x; word < words_per_hop; word += blockDim.x) {
                    reachable[word] = 0;
                }
                __syncthreads();
                has_path = true;
            }
            if (!threadIdx.x) {
                reachable[i / kParentWordBits] |= uint32_t{1} << (i % kParentWordBits);
            }
        }
        if (has_path) {
            // Compact completed hops in place. Later hops are still unread.
            if (!threadIdx.x)
                first_hops[witness_count] = hop;
            __syncthreads();
            ++witness_count;
        }
    }
    return false;
}

// Commit in packed (c, a) order. A new tail can add a parent for a later edge.
__global__ void r10_apply_kernel(int8_t* pag, int p, const int32_t* unf, int U, uint8_t* visited,
                                 int* q_node, int* q_pred, int qmax, int* path, int* lst,
                                 int* parents, int* first_hops, const int* r10_a, const int* r10_c,
                                 int n_r10, uint32_t* parents_by_hop) {
    const Marks graph{pag, p};
    const PathScratch work{visited, q_node, q_pred, qmax, path, lst};
    for (int edge = 0; edge < n_r10; ++edge) {
        const int start = r10_a[edge], end = r10_c[edge];
        if (graph(end, start) != 1)
            continue;
        const int count = compact_nodes(pag, nullptr, p, parents, Neighbors::parent, end, -1, -1);
        if (count < 2)
            continue;
        bool orient = r10_has_direct_witness(graph, start, parents, count, unf, U);
        if (!orient) {
            const int hops =
                compact_nodes(pag, nullptr, p, first_hops, Neighbors::first_hop, start, -1, end);
            if (hops < 2)
                continue;
            orient = r10_has_long_witness(graph, start, parents, count, first_hops, hops,
                                          parents_by_hop, unf, U, work);
        }
        if (orient && !threadIdx.x)
            put(pag, p, end, start, 3);
        __syncthreads();
    }
}

} // namespace

void launch_r1_packed(torch::Tensor pag, const int32_t* unf_ptr, int U, cudaStream_t stream) {
    const int p = static_cast<int>(pag.size(0));
    if (p == 0) {
        return;
    }
    const int t = kBlock;
    const int g = (p + t - 1) / t;
    auto counts = torch::empty({p}, pag.options().dtype(at::kInt));
    pack_r1_count_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, counts.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    auto off = torch::empty({p + 1}, pag.options().dtype(at::kInt));
    prefix_offsets(counts, off);
    const int n = off[p].item<int>();
    if (n <= 0) {
        return;
    }
    auto r1_a = torch::empty({n}, pag.options().dtype(at::kInt));
    auto r1_b = torch::empty({n}, pag.options().dtype(at::kInt));
    pack_r1_fill_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, off.data_ptr<int>(),
                                             r1_a.data_ptr<int>(), r1_b.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    r1_kernel<<<1, kBlock, 0, stream>>>(pag.data_ptr<int8_t>(), p, unf_ptr, U, r1_a.data_ptr<int>(),
                                        r1_b.data_ptr<int>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor> pack_r1_cuda(torch::Tensor pag) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    const int p = static_cast<int>(pag.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int t = kBlock;
    const int g = p == 0 ? 1 : (p + t - 1) / t;
    auto counts = torch::empty({p}, pag.options().dtype(at::kInt));
    if (p) {
        pack_r1_count_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p,
                                                  counts.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto off = torch::empty({p + 1}, pag.options().dtype(at::kInt));
    prefix_offsets(counts, off);
    const int n = p ? off[p].item<int>() : 0;
    auto r1_a = torch::empty({n}, pag.options().dtype(at::kInt));
    auto r1_b = torch::empty({n}, pag.options().dtype(at::kInt));
    if (n > 0) {
        pack_r1_fill_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, off.data_ptr<int>(),
                                                 r1_a.data_ptr<int>(), r1_b.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {r1_a, r1_b};
}

void pag_r1_cuda(torch::Tensor pag, torch::Tensor unf) {
    TORCH_CHECK(pag.is_cuda(), "pag R1 must be CUDA");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag must be square");
    TORCH_CHECK(pag.scalar_type() == at::kChar, "pag must be int8");
    TORCH_CHECK(pag.is_contiguous(), "pag must be contiguous");
    int U = 0;
    const int32_t* unf_ptr = nullptr;
    if (unf.defined() && unf.numel() > 0) {
        TORCH_CHECK(unf.is_cuda() && unf.scalar_type() == at::kInt && unf.is_contiguous(), "unf");
        TORCH_CHECK(unf.dim() == 2 && unf.size(1) == 3, "unf must be (U,3)");
        U = static_cast<int>(unf.size(0));
        unf_ptr = unf.data_ptr<int32_t>();
    }
    if (pag.size(0) == 0) {
        return;
    }
    launch_r1_packed(pag, unf_ptr, U, at::cuda::getCurrentCUDAStream());
}

void launch_r2_packed(torch::Tensor pag, cudaStream_t stream) {
    const int p = static_cast<int>(pag.size(0));
    if (p == 0) {
        return;
    }
    const int t = kBlock;
    const int g = (p + t - 1) / t;
    auto counts = torch::empty({p}, pag.options().dtype(at::kInt));
    pack_r2_count_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, counts.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    auto off = torch::empty({p + 1}, pag.options().dtype(at::kInt));
    prefix_offsets(counts, off);
    const int n = off[p].item<int>();
    if (n <= 0) {
        return;
    }
    auto r2_a = torch::empty({n}, pag.options().dtype(at::kInt));
    auto r2_c = torch::empty({n}, pag.options().dtype(at::kInt));
    pack_r2_fill_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, off.data_ptr<int>(),
                                             r2_a.data_ptr<int>(), r2_c.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    r2_kernel<<<1, kBlock, 0, stream>>>(pag.data_ptr<int8_t>(), p, r2_a.data_ptr<int>(),
                                        r2_c.data_ptr<int>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor> pack_r2_cuda(torch::Tensor pag) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    const int p = static_cast<int>(pag.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int t = kBlock;
    const int g = p == 0 ? 1 : (p + t - 1) / t;
    auto counts = torch::empty({p}, pag.options().dtype(at::kInt));
    if (p) {
        pack_r2_count_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p,
                                                  counts.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto off = torch::empty({p + 1}, pag.options().dtype(at::kInt));
    prefix_offsets(counts, off);
    const int n = p ? off[p].item<int>() : 0;
    auto r2_a = torch::empty({n}, pag.options().dtype(at::kInt));
    auto r2_c = torch::empty({n}, pag.options().dtype(at::kInt));
    if (n > 0) {
        pack_r2_fill_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, off.data_ptr<int>(),
                                                 r2_a.data_ptr<int>(), r2_c.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {r2_a, r2_c};
}

void pag_r2_cuda(torch::Tensor pag) {
    TORCH_CHECK(pag.is_cuda(), "pag R2 must be CUDA");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag must be square");
    TORCH_CHECK(pag.scalar_type() == at::kChar, "pag must be int8");
    TORCH_CHECK(pag.is_contiguous(), "pag must be contiguous");
    if (pag.size(0) == 0) {
        return;
    }
    launch_r2_packed(pag, at::cuda::getCurrentCUDAStream());
}

int pack_r3_into(torch::Tensor pag, cudaStream_t stream, torch::Tensor& r_b, torch::Tensor& r_d) {
    const int p = static_cast<int>(pag.size(0));
    if (p == 0) {
        r_b = torch::empty({0}, pag.options().dtype(at::kInt));
        r_d = torch::empty({0}, pag.options().dtype(at::kInt));
        return 0;
    }
    const int t = kBlock;
    const int g = (p + t - 1) / t;
    auto counts = torch::empty({p}, pag.options().dtype(at::kInt));
    pack_r3_count_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, counts.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    auto off = torch::empty({p + 1}, pag.options().dtype(at::kInt));
    prefix_offsets(counts, off);
    const int n = off[p].item<int>();
    r_b = torch::empty({n}, pag.options().dtype(at::kInt));
    r_d = torch::empty({n}, pag.options().dtype(at::kInt));
    if (n > 0) {
        pack_r3_fill_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, off.data_ptr<int>(),
                                                 r_b.data_ptr<int>(), r_d.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return n;
}

int pack_r5_into(torch::Tensor pag, cudaStream_t stream, torch::Tensor& r_a, torch::Tensor& r_b) {
    const int p = static_cast<int>(pag.size(0));
    if (p == 0) {
        r_a = torch::empty({0}, pag.options().dtype(at::kInt));
        r_b = torch::empty({0}, pag.options().dtype(at::kInt));
        return 0;
    }
    const int t = kBlock;
    const int g = (p + t - 1) / t;
    auto counts = torch::empty({p}, pag.options().dtype(at::kInt));
    pack_r5_count_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, counts.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    auto off = torch::empty({p + 1}, pag.options().dtype(at::kInt));
    prefix_offsets(counts, off);
    const int n = off[p].item<int>();
    r_a = torch::empty({n}, pag.options().dtype(at::kInt));
    r_b = torch::empty({n}, pag.options().dtype(at::kInt));
    if (n > 0) {
        pack_r5_fill_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, off.data_ptr<int>(),
                                                 r_a.data_ptr<int>(), r_b.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return n;
}

int pack_r8_into(torch::Tensor pag, cudaStream_t stream, torch::Tensor& r_a, torch::Tensor& r_c) {
    const int p = static_cast<int>(pag.size(0));
    if (p == 0) {
        r_a = torch::empty({0}, pag.options().dtype(at::kInt));
        r_c = torch::empty({0}, pag.options().dtype(at::kInt));
        return 0;
    }
    const int t = kBlock;
    const int g = (p + t - 1) / t;
    auto counts = torch::empty({p}, pag.options().dtype(at::kInt));
    pack_r8_count_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, counts.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    auto off = torch::empty({p + 1}, pag.options().dtype(at::kInt));
    prefix_offsets(counts, off);
    const int n = off[p].item<int>();
    r_a = torch::empty({n}, pag.options().dtype(at::kInt));
    r_c = torch::empty({n}, pag.options().dtype(at::kInt));
    if (n > 0) {
        pack_r8_fill_kernel<<<g, t, 0, stream>>>(pag.data_ptr<int8_t>(), p, off.data_ptr<int>(),
                                                 r_a.data_ptr<int>(), r_c.data_ptr<int>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return n;
}

void launch_r3_packed(torch::Tensor pag, const int32_t* unf_ptr, int U, cudaStream_t stream) {
    torch::Tensor r3_b;
    torch::Tensor r3_d;
    const int n = pack_r3_into(pag, stream, r3_b, r3_d);
    if (n <= 0) {
        return;
    }
    r3_kernel<<<1, kBlock, 0, stream>>>(pag.data_ptr<int8_t>(), static_cast<int>(pag.size(0)),
                                        unf_ptr, U, r3_b.data_ptr<int>(), r3_d.data_ptr<int>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor> pack_r3_cuda(torch::Tensor pag) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    torch::Tensor r3_b;
    torch::Tensor r3_d;
    pack_r3_into(pag, at::cuda::getCurrentCUDAStream(), r3_b, r3_d);
    return {r3_b, r3_d};
}

void pag_r3_cuda(torch::Tensor pag, torch::Tensor unf) {
    TORCH_CHECK(pag.is_cuda(), "pag R3 must be CUDA");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag must be square");
    TORCH_CHECK(pag.scalar_type() == at::kChar, "pag must be int8");
    TORCH_CHECK(pag.is_contiguous(), "pag must be contiguous");
    int U = 0;
    const int32_t* unf_ptr = nullptr;
    if (unf.defined() && unf.numel() > 0) {
        TORCH_CHECK(unf.is_cuda() && unf.scalar_type() == at::kInt && unf.is_contiguous(), "unf");
        TORCH_CHECK(unf.dim() == 2 && unf.size(1) == 3, "unf must be (U,3)");
        U = static_cast<int>(unf.size(0));
        unf_ptr = unf.data_ptr<int32_t>();
    }
    if (pag.size(0) == 0) {
        return;
    }
    launch_r3_packed(pag, unf_ptr, U, at::cuda::getCurrentCUDAStream());
}

void bind_sep(torch::Tensor S, torch::Tensor ranges, int p, const char* stage,
              const int64_t*& S_ptr, const int64_t*& ranges_ptr) {
    require_sepset_csr_cuda(S, ranges, p, stage);
    S_ptr = S.defined() ? S.data_ptr<int64_t>() : nullptr;
    ranges_ptr = ranges.defined() ? ranges.data_ptr<int64_t>() : nullptr;
}

void bind_unf(torch::Tensor unf, const int32_t*& unf_ptr, int& U) {
    U = 0;
    unf_ptr = nullptr;
    if (unf.defined() && unf.numel() > 0) {
        TORCH_CHECK(unf.is_cuda() && unf.scalar_type() == at::kInt && unf.is_contiguous(), "unf");
        TORCH_CHECK(unf.dim() == 2 && unf.size(1) == 3, "unf (U,3)");
        U = static_cast<int>(unf.size(0));
        unf_ptr = unf.data_ptr<int32_t>();
    }
}

void launch_r4_apply(torch::Tensor pag, const int64_t* S_ptr, const int64_t* ranges_ptr,
                     const int* r_b, const int* r_c, int n, cudaStream_t stream) {
    const int p = static_cast<int>(pag.size(0));
    const int qmax = p * p > 0 ? p * p : 1;
    auto visited = torch::empty({p}, pag.options().dtype(at::kByte));
    auto q_node = torch::empty({qmax}, pag.options().dtype(at::kInt));
    auto q_pred = torch::empty({qmax}, pag.options().dtype(at::kInt));
    auto path = torch::empty({p + 4}, pag.options().dtype(at::kInt));
    auto lst = torch::empty({p}, pag.options().dtype(at::kInt));
    auto indA = torch::empty({p}, pag.options().dtype(at::kInt));
    auto leak = torch::full({1}, -1, pag.options().dtype(at::kInt));
    r4_apply_kernel<<<1, kBlock, 0, stream>>>(
        pag.data_ptr<int8_t>(), p, S_ptr, ranges_ptr, visited.data_ptr<uint8_t>(),
        q_node.data_ptr<int>(), q_pred.data_ptr<int>(), qmax, path.data_ptr<int>(),
        lst.data_ptr<int>(), indA.data_ptr<int>(), r_b, r_c, n, leak.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_r4_packed(torch::Tensor pag, const int64_t* S_ptr, const int64_t* ranges_ptr,
                      cudaStream_t stream) {
    torch::Tensor r_b;
    torch::Tensor r_c;
    const int n = pack_r3_into(pag, stream, r_b, r_c);
    if (n <= 0) {
        return;
    }
    launch_r4_apply(pag, S_ptr, ranges_ptr, r_b.data_ptr<int>(), r_c.data_ptr<int>(), n, stream);
}

void launch_r5_apply(torch::Tensor pag, const int32_t* unf_ptr, int U, const int* r_a,
                     const int* r_b, int n, cudaStream_t stream) {
    const int p = static_cast<int>(pag.size(0));
    const int qmax = p * p > 0 ? p * p : 1;
    auto visited = torch::empty({p}, pag.options().dtype(at::kByte));
    auto q_node = torch::empty({qmax}, pag.options().dtype(at::kInt));
    auto q_pred = torch::empty({qmax}, pag.options().dtype(at::kInt));
    auto path = torch::empty({p + 4}, pag.options().dtype(at::kInt));
    auto lst = torch::empty({p}, pag.options().dtype(at::kInt));
    auto indC = torch::empty({p}, pag.options().dtype(at::kInt));
    auto indD = torch::empty({p}, pag.options().dtype(at::kInt));
    auto leak = torch::full({1}, -1, pag.options().dtype(at::kInt));
    r5_apply_kernel<<<1, kBlock, 0, stream>>>(
        pag.data_ptr<int8_t>(), p, unf_ptr, U, visited.data_ptr<uint8_t>(), q_node.data_ptr<int>(),
        q_pred.data_ptr<int>(), qmax, path.data_ptr<int>(), lst.data_ptr<int>(),
        indC.data_ptr<int>(), indD.data_ptr<int>(), r_a, r_b, n, leak.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_r5_packed(torch::Tensor pag, const int32_t* unf_ptr, int U, cudaStream_t stream) {
    torch::Tensor r_a;
    torch::Tensor r_b;
    const int n = pack_r5_into(pag, stream, r_a, r_b);
    if (n <= 0) {
        return;
    }
    launch_r5_apply(pag, unf_ptr, U, r_a.data_ptr<int>(), r_b.data_ptr<int>(), n, stream);
}

void launch_r6_packed(torch::Tensor pag, cudaStream_t stream) {
    torch::Tensor r_b;
    torch::Tensor r_c;
    const int n = pack_r3_into(pag, stream, r_b, r_c);
    if (n <= 0) {
        return;
    }
    r6_kernel<<<1, kBlock, 0, stream>>>(pag.data_ptr<int8_t>(), static_cast<int>(pag.size(0)),
                                        r_b.data_ptr<int>(), r_c.data_ptr<int>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_r7_packed(torch::Tensor pag, const int32_t* unf_ptr, int U, cudaStream_t stream) {
    torch::Tensor r_b;
    torch::Tensor r_c;
    const int n = pack_r3_into(pag, stream, r_b, r_c);
    if (n <= 0) {
        return;
    }
    r7_kernel<<<1, kBlock, 0, stream>>>(pag.data_ptr<int8_t>(), static_cast<int>(pag.size(0)),
                                        unf_ptr, U, r_b.data_ptr<int>(), r_c.data_ptr<int>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_r8_packed(torch::Tensor pag, cudaStream_t stream) {
    torch::Tensor r_a;
    torch::Tensor r_c;
    const int n = pack_r8_into(pag, stream, r_a, r_c);
    if (n <= 0) {
        return;
    }
    r8_kernel<<<1, kBlock, 0, stream>>>(pag.data_ptr<int8_t>(), static_cast<int>(pag.size(0)),
                                        r_a.data_ptr<int>(), r_c.data_ptr<int>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_r9_apply(torch::Tensor pag, const int32_t* unf_ptr, int U, const int* r_a,
                     const int* r_c, int n, cudaStream_t stream) {
    const int p = static_cast<int>(pag.size(0));
    const int qmax = p * p > 0 ? p * p : 1;
    auto visited = torch::empty({p}, pag.options().dtype(at::kByte));
    auto q_node = torch::empty({qmax}, pag.options().dtype(at::kInt));
    auto q_pred = torch::empty({qmax}, pag.options().dtype(at::kInt));
    auto path = torch::empty({p + 4}, pag.options().dtype(at::kInt));
    auto lst = torch::empty({p}, pag.options().dtype(at::kInt));
    auto indB = torch::empty({p}, pag.options().dtype(at::kInt));
    auto leak = torch::full({1}, -1, pag.options().dtype(at::kInt));
    r9_apply_kernel<<<1, kBlock, 0, stream>>>(
        pag.data_ptr<int8_t>(), p, unf_ptr, U, visited.data_ptr<uint8_t>(), q_node.data_ptr<int>(),
        q_pred.data_ptr<int>(), qmax, path.data_ptr<int>(), lst.data_ptr<int>(),
        indB.data_ptr<int>(), r_a, r_c, n, leak.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_r9_packed(torch::Tensor pag, const int32_t* unf_ptr, int U, cudaStream_t stream) {
    torch::Tensor r_a;
    torch::Tensor r_c;
    const int n = pack_r8_into(pag, stream, r_a, r_c);
    if (n <= 0) {
        return;
    }
    launch_r9_apply(pag, unf_ptr, U, r_a.data_ptr<int>(), r_c.data_ptr<int>(), n, stream);
}

void launch_r10_apply(torch::Tensor pag, const int32_t* unf_ptr, int U, const int* r_a,
                      const int* r_c, int n, cudaStream_t stream) {
    const int p = static_cast<int>(pag.size(0));
    const int qmax = p * p > 0 ? p * p : 1;
    auto visited = torch::empty({p}, pag.options().dtype(at::kByte));
    auto q_node = torch::empty({qmax}, pag.options().dtype(at::kInt));
    auto q_pred = torch::empty({qmax}, pag.options().dtype(at::kInt));
    auto path = torch::empty({p + 4}, pag.options().dtype(at::kInt));
    auto lst = torch::empty({p}, pag.options().dtype(at::kInt));
    auto indB = torch::empty({p}, pag.options().dtype(at::kInt));
    auto indX = torch::empty({p}, pag.options().dtype(at::kInt));
    const int64_t words = (static_cast<int64_t>(p) + kParentWordBits - 1) / kParentWordBits;
    auto parents_by_hop = torch::empty({p, words}, pag.options().dtype(at::kInt));
    r10_apply_kernel<<<1, kBlock, 0, stream>>>(
        pag.data_ptr<int8_t>(), p, unf_ptr, U, visited.data_ptr<uint8_t>(), q_node.data_ptr<int>(),
        q_pred.data_ptr<int>(), qmax, path.data_ptr<int>(), lst.data_ptr<int>(),
        indB.data_ptr<int>(), indX.data_ptr<int>(), r_a, r_c, n,
        reinterpret_cast<uint32_t*>(parents_by_hop.data_ptr<int32_t>()));
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_r10_packed(torch::Tensor pag, const int32_t* unf_ptr, int U, cudaStream_t stream) {
    torch::Tensor r_a;
    torch::Tensor r_c;
    const int n = pack_r8_into(pag, stream, r_a, r_c);
    if (n <= 0) {
        return;
    }
    launch_r10_apply(pag, unf_ptr, U, r_a.data_ptr<int>(), r_c.data_ptr<int>(), n, stream);
}

std::tuple<torch::Tensor, torch::Tensor> pack_r8_cuda(torch::Tensor pag) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    torch::Tensor r_a;
    torch::Tensor r_c;
    pack_r8_into(pag, at::cuda::getCurrentCUDAStream(), r_a, r_c);
    return {r_a, r_c};
}

std::tuple<torch::Tensor, torch::Tensor> pack_r5_cuda(torch::Tensor pag) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    torch::Tensor r_a;
    torch::Tensor r_b;
    pack_r5_into(pag, at::cuda::getCurrentCUDAStream(), r_a, r_b);
    return {r_a, r_b};
}

void pag_r4_cuda(torch::Tensor pag, torch::Tensor S, torch::Tensor ranges) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    const int64_t* S_ptr = nullptr;
    const int64_t* ranges_ptr = nullptr;
    bind_sep(S, ranges, static_cast<int>(pag.size(0)), "pag_r4", S_ptr, ranges_ptr);
    if (pag.size(0) == 0) {
        return;
    }
    launch_r4_packed(pag, S_ptr, ranges_ptr, at::cuda::getCurrentCUDAStream());
}

void pag_r5_cuda(torch::Tensor pag, torch::Tensor unf) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    int U = 0;
    const int32_t* unf_ptr = nullptr;
    bind_unf(unf, unf_ptr, U);
    if (pag.size(0) == 0) {
        return;
    }
    launch_r5_packed(pag, unf_ptr, U, at::cuda::getCurrentCUDAStream());
}

void pag_r6_cuda(torch::Tensor pag) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    if (pag.size(0) == 0) {
        return;
    }
    launch_r6_packed(pag, at::cuda::getCurrentCUDAStream());
}

void pag_r9_cuda(torch::Tensor pag, torch::Tensor unf) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    int U = 0;
    const int32_t* unf_ptr = nullptr;
    bind_unf(unf, unf_ptr, U);
    if (pag.size(0) == 0) {
        return;
    }
    launch_r9_packed(pag, unf_ptr, U, at::cuda::getCurrentCUDAStream());
}

void pag_r7_cuda(torch::Tensor pag, torch::Tensor unf) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    int U = 0;
    const int32_t* unf_ptr = nullptr;
    bind_unf(unf, unf_ptr, U);
    if (pag.size(0) == 0) {
        return;
    }
    launch_r7_packed(pag, unf_ptr, U, at::cuda::getCurrentCUDAStream());
}

void pag_r8_cuda(torch::Tensor pag) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    if (pag.size(0) == 0) {
        return;
    }
    launch_r8_packed(pag, at::cuda::getCurrentCUDAStream());
}

void pag_r10_cuda(torch::Tensor pag, torch::Tensor unf) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    int U = 0;
    const int32_t* unf_ptr = nullptr;
    bind_unf(unf, unf_ptr, U);
    if (pag.size(0) == 0) {
        return;
    }
    launch_r10_packed(pag, unf_ptr, U, at::cuda::getCurrentCUDAStream());
}

torch::Tensor pag_orient_cuda(torch::Tensor pag, torch::Tensor unf, torch::Tensor S,
                              torch::Tensor ranges, torch::Tensor rules) {
    TORCH_CHECK(pag.is_cuda() && pag.is_contiguous() && pag.scalar_type() == at::kChar, "pag");
    TORCH_CHECK(pag.dim() == 2 && pag.size(0) == pag.size(1), "pag square");
    TORCH_CHECK(rules.is_cuda() && rules.is_contiguous() && rules.numel() >= 10, "rules");
    const int p = static_cast<int>(pag.size(0));
    auto times = torch::zeros({10}, torch::dtype(torch::kFloat64).device(torch::kCPU));
    if (p == 0) {
        return times;
    }
    int U = 0;
    const int32_t* unf_ptr = nullptr;
    if (unf.defined() && unf.numel() > 0) {
        TORCH_CHECK(unf.is_cuda() && unf.scalar_type() == at::kInt && unf.is_contiguous(), "unf");
        TORCH_CHECK(unf.dim() == 2 && unf.size(1) == 3, "unf (U,3)");
        U = static_cast<int>(unf.size(0));
        unf_ptr = unf.data_ptr<int32_t>();
    }
    require_sepset_csr_cuda(S, ranges, p, "pag_orient");
    const int64_t* S_ptr = (S.defined() && S.numel() > 0) ? S.data_ptr<int64_t>() : nullptr;
    const int64_t* ranges_ptr =
        (ranges.defined() && ranges.numel() > 0) ? ranges.data_ptr<int64_t>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const uint8_t* ru = rules.data_ptr<uint8_t>();
    auto hrules = rules.cpu();
    const uint8_t* hr = hrules.data_ptr<uint8_t>();
    double* ts = times.data_ptr<double>();

    auto stamp = [&](int i, auto&& launch) {
        const auto t0 = std::chrono::steady_clock::now();
        launch();
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        cudaStreamSynchronize(stream);
        ts[i] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    if (hr[0]) {
        stamp(0, [&] { launch_r1_packed(pag, unf_ptr, U, stream); });
    }
    if (hr[1]) {
        stamp(1, [&] { launch_r2_packed(pag, stream); });
    }
    if (hr[2]) {
        stamp(2, [&] { launch_r3_packed(pag, unf_ptr, U, stream); });
    }
    if (hr[3]) {
        stamp(3, [&] { launch_r4_packed(pag, S_ptr, ranges_ptr, stream); });
    }
    if (hr[4]) {
        stamp(4, [&] { launch_r5_packed(pag, unf_ptr, U, stream); });
    }
    if (hr[5]) {
        stamp(5, [&] { launch_r6_packed(pag, stream); });
    }
    if (hr[6]) {
        stamp(6, [&] { launch_r7_packed(pag, unf_ptr, U, stream); });
    }
    if (hr[7]) {
        stamp(7, [&] { launch_r8_packed(pag, stream); });
    }
    if (hr[8]) {
        stamp(8, [&] { launch_r9_packed(pag, unf_ptr, U, stream); });
    }
    if (hr[9]) {
        stamp(9, [&] { launch_r10_packed(pag, unf_ptr, U, stream); });
    }
    (void)ru;
    return times;
}

void udag2pag_preload() {
    cuda_preload_context();
    touch_cuda_kernel(pack_r1_count_kernel);
    touch_cuda_kernel(pack_r1_fill_kernel);
    touch_cuda_kernel(r1_kernel);
    touch_cuda_kernel(pack_r2_count_kernel);
    touch_cuda_kernel(pack_r2_fill_kernel);
    touch_cuda_kernel(r2_kernel);
    touch_cuda_kernel(pack_r3_count_kernel);
    touch_cuda_kernel(pack_r3_fill_kernel);
    touch_cuda_kernel(r3_kernel);
    touch_cuda_kernel(pack_r5_count_kernel);
    touch_cuda_kernel(pack_r5_fill_kernel);
    touch_cuda_kernel(r4_apply_kernel);
    touch_cuda_kernel(r5_apply_kernel);
    touch_cuda_kernel(r6_kernel);
    touch_cuda_kernel(pack_r8_count_kernel);
    touch_cuda_kernel(pack_r8_fill_kernel);
    touch_cuda_kernel(r7_kernel);
    touch_cuda_kernel(r8_kernel);
    touch_cuda_kernel(r9_apply_kernel);
    touch_cuda_kernel(r10_apply_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("r1", &pag_r1_cuda);
    m.def("pack_r1", &pack_r1_cuda);
    m.def("r2", &pag_r2_cuda);
    m.def("pack_r2", &pack_r2_cuda);
    m.def("r3", &pag_r3_cuda);
    m.def("pack_r3", &pack_r3_cuda);
    m.def("r4", &pag_r4_cuda);
    m.def("r5", &pag_r5_cuda);
    m.def("pack_r5", &pack_r5_cuda);
    m.def("r6", &pag_r6_cuda);
    m.def("r7", &pag_r7_cuda);
    m.def("r8", &pag_r8_cuda);
    m.def("pack_r8", &pack_r8_cuda);
    m.def("r9", &pag_r9_cuda);
    m.def("r10", &pag_r10_cuda);
    m.def("orient", &pag_orient_cuda);
    m.def("preload", &udag2pag_preload);
}
