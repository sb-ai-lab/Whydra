// ---------------------------------------------------------------------------
// graph_batch.cu — CUDA batch builder for PC/FCI conditional-independence
// (CI) testing, per-depth drivers, and separating-set (sepset) bookkeeping.
//
// PURPOSE
//   Given a correlation/covariance matrix C and an adjacency mask G, enumerate
//   for each depth k every conditioning set S of size k drawn from the packed
//   neighbour lists, compute Fisher-Z r(x, y | S) for each (x, y, S), and
//   record the separating sets that make x and y independent. The PC-stable and
//   FCI skeleton paths keep G frozen during a depth and accumulate the UNION of
//   all separating S per pair (FCI then mirrors the record last-writer-wins);
//   the possible-D-SEP path records the FIRST hit with sequential last-writer
//   semantics.
//
// PYTHON COUNTERPART
//   whydra/algorithms/graph_core/cuda_graph_batch.py — ``run_skeleton`` /
//   ``run_depth`` / ``SepsetCSR`` and the wrapper for every binding below.
//   The k<=3 unrolled LDLT math also exists in ``fisher_z_ldlt.cu``; here it
//   is duplicated (and kept in sync) as ``r_cond``/``r_cond_generic`` so a
//   depth needs no Python round trip. Tests: ``tests/test_cuda_pc_sepset_union.py``.
//
// MAIN ENTRY POINTS (all exported by PYBIND11_MODULE at the end)
//   pc_skeleton_union  whole PC-stable skeleton loop (depth loop stays in C++)
//   pc_depth_union     one PC-stable depth: live pairs -> fused CI -> commit
//   pc_live_pairs      live directed pairs + per-pair test count C(deg-1, k)
//   ci_from_comb       flat combinadic CI batch (generic k / FCI path)
//   claim_sepsets      first-hit sepset CSR claim (FCI / possible-D-SEP)
//   commit_union_*     PC-stable union sepsets (mark, then write)
//   pdsep_commit_walk  last-writer sepset commit for possible-D-SEP
//   graph_batch_*      primitive pieces (degrees, compact, fill, hits, ...)
//
// MEMORY / LAYOUT MODEL
//   * G is int8 [p, p] with arbitrary strides (gs0, gs1); 0 = edge absent.
//     An undirected skeleton mirrors both triangles. During a PC depth G is
//     frozen and only the hit arcs are zeroed at commit time.
//   * nbrs is int32 [p, max_d] packed in INCREASING index order so membership
//     is a binary search; deg is int32 [p]. y is omitted from the candidate
//     list of x, hence C(deg(x)-1, k) tests per directed pair.
//   * Tests of one pair are ordered by leaf combinadic rank of
//     combinations(nbrs(x)\{y}, k) — the same order as itertools.combinations
//     / np.delete(np.where(G[x] != 0), y).
//   * Sepsets form a CSR: ranges[p, p, 2] = (start, end) into a 1-D int64 S;
//     (-1, -1) = edge still present, (0, 0) = empty sepset (k = 0).
//   * Pair-major kernels launch one CTA per live directed pair; flat kernels
//     launch one thread per test and binary-search ``offsets`` for its pair.
//
// SEARCH RULES (see tests/test_cuda_pc_sepset_union.py)
//   * Fisher-Z compares the statistic with the host threshold: < for
//     PC-stable (p > alpha), <= for FCI and Possible-D-SEP (p >= alpha).
//     Rounding can change boundary decisions, and very small alpha can
//     produce an infinite threshold.
//   * PC-stable records the UNION of every separating S's members per pair
//     and writes one record to BOTH ranges cells. FCI records the first S per
//     directed pair; the higher endpoint wins when both directions hit.
//   * k = 0 CI is symmetric: the workspace path enumerates x < y only and
//     mirrors the empty sepset. k >= 1 stays directed because candidate lists
//     and sepset records are per directed pair.
//   * The depth drivers clear both triangles (G[x,y] and G[y,x]) for every
//     removed edge; individual kernels may receive one directed cell.
//
// RUNTIME SWITCHES (environment; default on, used for A/B validation)
//   WHYDRA_PC_SMALLK=0     flat ci_from_comb instead of pair-major k <= 3
//   WHYDRA_PC_WORKSPACE=0  per-depth allocations instead of cached workspace
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_check.cuh"
#include <cmath>
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <unordered_map>
#include "cuda_float.cuh"
#include "cuda_touch.cuh"

namespace {

__device__ inline void atomic_min_abs(unsigned long long* addr, double r) {
    if (!isfinite(r)) {
        return;
    }
    const unsigned long long bits = __double_as_longlong(fabs(r));
    atomicMin(addr, bits);
}

constexpr int kBlock = 256;
constexpr int kMaxFused = 32; // fused CI cap (registers/smem budget)
constexpr int kMaxCond = 96;  // generic conditioning-set cap
// |r| is clamped away from 1 so the Fisher-Z transform stays finite.
constexpr double kCutAt = 0.9999999;

// ===========================================================================
// Index algebra: saturating binomials, sorted-neighbour search, combinadic
// rank/unrank, and 64-bit atomics. Integer-only; shared by every CI kernel.
// The rank order is lexicographic (leaf order) to match
// itertools.combinations, which the CPU reference uses.
// ===========================================================================

// C(n, k) saturating at -1 on int64 overflow (callers treat -1 as "too
// many"). Used both for per-pair test counts and for combinadic binom terms.
__device__ __forceinline__ int64_t binom_sat(int n, int k) {
    if (k < 0 || n < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n - k) {
        k = n - k;
    }
    int64_t r = 1;
    constexpr int64_t kMaxI64 = 9223372036854775807LL;
    for (int i = 1; i <= k; ++i) {
        int64_t m = static_cast<int64_t>(n - k + i);
        if (m > 0 && r > kMaxI64 / m) {
            return -1;
        }
        r *= m;
        r /= i;
    }
    return r;
}

__device__ __forceinline__ int find_y_idx(const int* row, int d, int y) {
    for (int j = 0; j < d; ++j) {
        if (row[j] == y) {
            return j;
        }
    }
    return -1;
}

// ``nbrs`` rows are packed in increasing index order, so the position of ``v``
// is a binary search (the linear ``find_y_idx`` is O(deg) and dominates k=1,
// where deg can be in the hundreds).
__device__ __forceinline__ int sorted_find_idx(const int* row, int d, int v) {
    int lo = 0, hi = d;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int mv = row[mid];
        if (mv == v) {
            return mid;
        }
        if (mv < v) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return -1;
}

// ===========================================================================
// Independence tests and partial correlations.
//   r_cond / r_cond_generic
//       Fisher-Z partial correlation r(x, y | S) via an unpivoted LDL^T of the
//       conditioning block. k = 0..3 are unrolled (registers only); the
//       generic path serves larger k. ``eps`` is the modified-Cholesky pivot
//       guard and is 0 on the exact path.
//   r_is_indep / p_from_r
//       Convert r to a z-statistic / two-sided p-value. PC-stable is STRICT
//       (p > alpha, ge_alpha = 0); FCI/possible-D-SEP is non-strict
//       (p >= alpha, ge_alpha = 1). NaN/inf r is never independent.
// ===========================================================================

__device__ __forceinline__ bool r_is_indep(double rv, int n_samples, int k, double tau,
                                           int ge_alpha) {
    if (!isfinite(rv)) {
        return false;
    }
    if (rv > kCutAt) {
        rv = kCutAt;
    } else if (rv < -kCutAt) {
        rv = -kCutAt;
    }
    double Z = 0.5 * log((1.0 + rv) / (1.0 - rv));
    double df = static_cast<double>(n_samples - k - 3);
    if (!(df > 0.0)) {
        return false;
    }
    double stat = sqrt(df) * fabs(Z);
    return ge_alpha ? (stat <= tau) : (stat < tau);
}

__device__ __forceinline__ double p_from_r(double rv, int n_samples, int k) {
    if (!isfinite(rv)) {
        return -INFINITY;
    }
    if (rv > kCutAt) {
        rv = kCutAt;
    } else if (rv < -kCutAt) {
        rv = -kCutAt;
    }
    double Z = 0.5 * log((1.0 + rv) / (1.0 - rv));
    double df = static_cast<double>(n_samples - k - 3);
    if (!(df > 0.0)) {
        return -INFINITY;
    }
    double stat = sqrt(df) * fabs(Z);
    return erfc(stat * 0.7071067811865476);
}

__device__ void atomic_max_f64(double* addr, double val) {
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *uaddr;
    unsigned long long assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) {
            return;
        }
        old = atomicCAS(uaddr, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

template <typename T> __device__ __forceinline__ T dsqrt_t(T x);

template <> __device__ __forceinline__ float dsqrt_t<float>(float x) { return sqrtf(x); }
template <> __device__ __forceinline__ double dsqrt_t<double>(double x) { return sqrt(x); }

template <typename T> __device__ __forceinline__ T dfabs_t(T x);

template <> __device__ __forceinline__ float dfabs_t<float>(float x) { return fabsf(x); }
template <> __device__ __forceinline__ double dfabs_t<double>(double x) { return fabs(x); }

template <typename T> __device__ __forceinline__ T r_from_dets_t(T qxy, T qxx, T qyy) {
    return qxy / dsqrt_t(dfabs_t(qxx * qyy));
}

template <typename T>
__device__ __forceinline__ T c_at(const T* C, int i, int j, long long s0, long long s1) {
    return __ldg(C + static_cast<long long>(i) * s0 + static_cast<long long>(j) * s1);
}

template <typename T>
__device__ T r_cond(const T* C, long long s0, long long s1, int x, int y, const int* S, int k,
                    T eps) {
    const T axx = c_at(C, x, x, s0, s1);
    const T ayy = c_at(C, y, y, s0, s1);
    const T axy = c_at(C, x, y, s0, s1);
    if (k <= 0) {
        return axy / dsqrt_t(dfabs_t(axx * ayy));
    }
    if (k == 1) {
        const int s = S[0];
        const T css = c_at(C, s, s, s0, s1);
        const T cxs = c_at(C, x, s, s0, s1);
        const T cys = c_at(C, y, s, s0, s1);
        const T inv = T(1) / (css + eps);
        return r_from_dets_t(axy - cxs * cys * inv, axx - cxs * cxs * inv, ayy - cys * cys * inv);
    }
    if (k == 2) {
        const int z0 = S[0];
        const int z1 = S[1];
        const T g00 = c_at(C, z0, z0, s0, s1);
        const T g01 = c_at(C, z0, z1, s0, s1);
        const T g11 = c_at(C, z1, z1, s0, s1);
        const T ax0 = c_at(C, z0, x, s0, s1);
        const T ax1 = c_at(C, z1, x, s0, s1);
        const T ay0 = c_at(C, z0, y, s0, s1);
        const T ay1 = c_at(C, z1, y, s0, s1);
        const T L10 = (g00 != T(0)) ? g01 / g00 : T(0);
        const T D1 = g11 - L10 * L10 * g00;
        const T ux1 = ax1 - L10 * ax0;
        const T uy1 = ay1 - L10 * ay0;
        const T inv0 = T(1) / (g00 + eps);
        const T inv1 = T(1) / (D1 + eps);
        return r_from_dets_t(axy - ax0 * ay0 * inv0 - ux1 * uy1 * inv1,
                             axx - ax0 * ax0 * inv0 - ux1 * ux1 * inv1,
                             ayy - ay0 * ay0 * inv0 - uy1 * uy1 * inv1);
    }
    const int z0 = S[0];
    const int z1 = S[1];
    const int z2 = S[2];
    const T g00 = c_at(C, z0, z0, s0, s1);
    const T g01 = c_at(C, z0, z1, s0, s1);
    const T g02 = c_at(C, z0, z2, s0, s1);
    const T g11 = c_at(C, z1, z1, s0, s1);
    const T g12 = c_at(C, z1, z2, s0, s1);
    const T g22 = c_at(C, z2, z2, s0, s1);
    const T ax0 = c_at(C, z0, x, s0, s1);
    const T ax1 = c_at(C, z1, x, s0, s1);
    const T ax2 = c_at(C, z2, x, s0, s1);
    const T ay0 = c_at(C, z0, y, s0, s1);
    const T ay1 = c_at(C, z1, y, s0, s1);
    const T ay2 = c_at(C, z2, y, s0, s1);
    const T L10 = (g00 != T(0)) ? g01 / g00 : T(0);
    const T L20 = (g00 != T(0)) ? g02 / g00 : T(0);
    const T D1 = g11 - L10 * L10 * g00;
    const T L21 = (D1 != T(0)) ? (g12 - L20 * L10 * g00) / D1 : T(0);
    const T D2 = g22 - L20 * L20 * g00 - L21 * L21 * D1;
    const T ux1 = ax1 - L10 * ax0;
    const T ux2 = ax2 - L20 * ax0 - L21 * ux1;
    const T uy1 = ay1 - L10 * ay0;
    const T uy2 = ay2 - L20 * ay0 - L21 * uy1;
    const T inv0 = T(1) / (g00 + eps);
    const T inv1 = T(1) / (D1 + eps);
    const T inv2 = T(1) / (D2 + eps);
    return r_from_dets_t(axy - ax0 * ay0 * inv0 - ux1 * uy1 * inv1 - ux2 * uy2 * inv2,
                         axx - ax0 * ax0 * inv0 - ux1 * ux1 * inv1 - ux2 * ux2 * inv2,
                         ayy - ay0 * ay0 * inv0 - uy1 * uy1 * inv1 - uy2 * uy2 * inv2);
}

template <typename T>
__device__ T r_cond_generic(const T* C, long long s0, long long s1, int x, int y, const int* S,
                            int k, T eps) {
    T L[kMaxFused * kMaxFused];
    T D[kMaxFused];
    T ux[kMaxFused];
    T uy[kMaxFused];
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            L[i * k + j] = c_at(C, S[i], S[j], s0, s1);
        }
        ux[i] = c_at(C, S[i], x, s0, s1);
        uy[i] = c_at(C, S[i], y, s0, s1);
    }
    for (int j = 0; j < k; ++j) {
        T acc = L[j * k + j];
        for (int p = 0; p < j; ++p) {
            const T lp = L[j * k + p];
            acc -= lp * lp * D[p];
        }
        D[j] = acc;
        L[j * k + j] = T(1);
        for (int i = j + 1; i < k; ++i) {
            T s = L[i * k + j];
            for (int p = 0; p < j; ++p) {
                s -= L[i * k + p] * L[j * k + p] * D[p];
            }
            L[i * k + j] = (acc != T(0)) ? s / acc : T(0);
        }
    }
    for (int i = 1; i < k; ++i) {
        T sx = ux[i];
        T sy = uy[i];
        for (int j = 0; j < i; ++j) {
            sx -= L[i * k + j] * ux[j];
            sy -= L[i * k + j] * uy[j];
        }
        ux[i] = sx;
        uy[i] = sy;
    }
    T qxy = c_at(C, x, y, s0, s1);
    T qxx = c_at(C, x, x, s0, s1);
    T qyy = c_at(C, y, y, s0, s1);
    for (int i = 0; i < k; ++i) {
        const T inv = T(1) / (D[i] + eps);
        qxy -= ux[i] * uy[i] * inv;
        qxx -= ux[i] * ux[i] * inv;
        qyy -= uy[i] * uy[i] * inv;
    }
    return r_from_dets_t(qxy, qxx, qyy);
}

__device__ void unrank_lex(int n, int k, int64_t idx, int* out) {
    int start = 0;
    for (int j = 0; j < k; ++j) {
        int remaining = k - j - 1;
        int x = start;
        int chosen = n - 1;
        for (; x < n; ++x) {
            int64_t n_combs = binom_sat(n - x - 1, remaining);
            if (n_combs < 0) {
                n_combs = 0;
            }
            if (idx < n_combs) {
                chosen = x;
                break;
            }
            idx -= n_combs;
        }
        out[j] = chosen;
        start = chosen + 1;
    }
}

__device__ int64_t pair_index(const int64_t* offsets, int64_t E, int64_t b) {
    int64_t lo = 0;
    int64_t hi = E;
    while (lo + 1 < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (offsets[mid] <= b) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ void atomic_min_i64(int64_t* addr, int64_t val) {
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *uaddr;
    unsigned long long assumed;
    do {
        assumed = old;
        if (static_cast<int64_t>(assumed) <= val) {
            return;
        }
        old = atomicCAS(uaddr, assumed, static_cast<unsigned long long>(val));
    } while (assumed != old);
}

__device__ __forceinline__ int64_t atomic_add_i64(int64_t* addr, int64_t val) {
    return static_cast<int64_t>(atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                                          static_cast<unsigned long long>(val)));
}

__device__ __forceinline__ int64_t atomic_cas_i64(int64_t* addr, int64_t cmp, int64_t val) {
    return static_cast<int64_t>(atomicCAS(reinterpret_cast<unsigned long long*>(addr),
                                          static_cast<unsigned long long>(cmp),
                                          static_cast<unsigned long long>(val)));
}

// ===========================================================================
// Adjacency compaction and the flat (offset-search) CI batch.
// These kernels back the generic k / FCI paths and the fallback driver:
// degrees -> compact -> emit_pairs -> count_tests -> prefix sum -> fill+CI.
// The pair-major PC fast path below replaces most of this for k <= 3.
// ===========================================================================

__global__ void degrees_kernel(const int8_t* __restrict__ G, int64_t s0, int64_t s1, int p,
                               int* __restrict__ deg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) {
        return;
    }
    int d = 0;
    for (int j = 0; j < p; ++j) {
        if (i == j) {
            continue;
        }
        if (G[i * s0 + j * s1] != 0) {
            ++d;
        }
    }
    deg[i] = d;
}

__global__ void compact_kernel(const int8_t* __restrict__ G, int64_t s0, int64_t s1, int p,
                               int max_d, int* __restrict__ nbrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) {
        return;
    }
    int w = 0;
    for (int j = 0; j < p; ++j) {
        if (i == j) {
            continue;
        }
        if (G[i * s0 + j * s1] != 0) {
            nbrs[i * max_d + w] = j;
            ++w;
        }
    }
}

__global__ void emit_pairs_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg,
                                  const int64_t* __restrict__ row_off, int p, int max_d,
                                  int* __restrict__ px, int* __restrict__ py) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= p) {
        return;
    }
    int64_t off = row_off[x];
    int d = deg[x];
    for (int j = 0; j < d; ++j) {
        px[off + j] = x;
        py[off + j] = nbrs[x * static_cast<int64_t>(max_d) + j];
    }
}

__global__ void count_tests_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg,
                                   const int* __restrict__ px, const int* __restrict__ py,
                                   int64_t E, int max_d, int k, int64_t* __restrict__ counts) {
    int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }
    int x = px[e];
    int y = py[e];
    int d = deg[x];
    const int* row = nbrs + static_cast<int64_t>(x) * max_d;
    int y_idx = sorted_find_idx(row, d, y);
    int n = d - (y_idx >= 0 ? 1 : 0);
    counts[e] = binom_sat(n, k);
}

__global__ void fill_batch_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg,
                                  const int* __restrict__ px, const int* __restrict__ py,
                                  const int64_t* __restrict__ offsets,
                                  const int64_t* __restrict__ rank_bases, int64_t E, int64_t B,
                                  int max_d, int k, int64_t rank_base, int* __restrict__ x_out,
                                  int* __restrict__ y_out, int* __restrict__ S) {
    int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }
    int64_t e = pair_index(offsets, E, b);
    int x = px[e];
    int y = py[e];
    x_out[b] = x;
    y_out[b] = y;
    if (k <= 0) {
        return;
    }
    int d = deg[x];
    const int* row = nbrs + static_cast<int64_t>(x) * max_d;
    int y_idx = sorted_find_idx(row, d, y);
    int n = d - (y_idx >= 0 ? 1 : 0);
    int64_t base = rank_bases != nullptr ? rank_bases[e] : rank_base;
    int64_t local = base + (b - offsets[e]);
    int combo[kMaxCond];
    unrank_lex(n, k, local, combo);
    for (int j = 0; j < k; ++j) {
        int pos = combo[j];
        if (y_idx >= 0 && pos >= y_idx) {
            ++pos;
        }
        S[b * k + j] = row[pos];
    }
}

// ===========================================================================
// Fused CI kernels: Fisher-Z + threshold + bookkeeping in one launch.
// ``kSMax`` is the compile-time cap that lets S live in registers: 4 for
// k <= 3 (fully unrolled r_cond) and kMaxFused for the generic kernel.
// ===========================================================================

template <typename T, int kSMax> struct fused_r {
    __device__ static T eval(const T* C, long long s0, long long s1, int x, int y, const int* S,
                             int k, T eps) {
        return r_cond(C, s0, s1, x, y, S, k, eps);
    }
};

template <typename T> struct fused_r<T, kMaxFused> {
    __device__ static T eval(const T* C, long long s0, long long s1, int x, int y, const int* S,
                             int k, T eps) {
        return r_cond_generic(C, s0, s1, x, y, S, k, eps);
    }
};

template <typename T, int kSMax>
__global__ void ci_from_comb_kernel(
    // One thread per test b. Grid ceil(B/256), block 256.
    // Integer: pair_index, unrank_lex. FP: fused LDLᵀ + z-threshold.
    // atomicMin first independent local rank. write_g zeros G (pdsep).
    const T* __restrict__ C, long long cs0, long long cs1, int8_t* __restrict__ G, int64_t gs0,
    int64_t gs1, const int* __restrict__ nbrs, const int* __restrict__ deg,
    const int* __restrict__ px, const int* __restrict__ py, const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ rank_bases, int64_t rank_base, int64_t* __restrict__ first_rank,
    int64_t E, int64_t B, int max_d, int k, int n_samples, double tau, int ge_alpha, int write_g,
    T eps, T* __restrict__ r_out, const int8_t* __restrict__ adj_x, int64_t adj_s0, int64_t adj_s1,
    uint8_t* __restrict__ union_marks, const int* __restrict__ rows, int64_t u_stride,
    uint8_t* __restrict__ hit_flags, unsigned long long* __restrict__ min_abs) {
    int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }
    int64_t e = pair_index(offsets, E, b);
    int x = px[e];
    int y = py[e];
    // FCI still needs r for prefix pMax of the reverse pair in the same
    // launch, so only skip a gone edge when r is unused. PC-stable keeps G
    // frozen until the end of the depth (write_g=0).
    if (write_g && G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] == 0 &&
        r_out == nullptr) {
        return;
    }
    int Sloc[kSMax];
    int64_t base = rank_bases != nullptr ? rank_bases[e] : rank_base;
    int64_t local = base + (b - offsets[e]);
    if (k > 0) {
        int d = deg[x];
        const int* row = nbrs + static_cast<int64_t>(x) * max_d;
        int y_idx;
        if (k == 1) {
            // Rank ``r`` of ``combinations(nbrs(x)\{y}, 1)`` is just the r-th
            // neighbor once ``y`` is skipped. Binary search for y's slot and
            // index directly — avoids the O(deg) find_y_idx + unrank_lex that
            // made k=1 memory-latency bound for high-degree nodes.
            y_idx = sorted_find_idx(row, d, y);
            int pos = static_cast<int>(local);
            if (y_idx >= 0 && pos >= y_idx) {
                ++pos;
            }
            Sloc[0] = row[pos];
        } else {
            y_idx = sorted_find_idx(row, d, y);
            int n = d - (y_idx >= 0 ? 1 : 0);
            int combo[kSMax];
            unrank_lex(n, k, local, combo);
            for (int j = 0; j < k; ++j) {
                int pos = combo[j];
                if (y_idx >= 0 && pos >= y_idx) {
                    ++pos;
                }
                Sloc[j] = row[pos];
            }
        }
        // Possible-D-SEP CI: skip S ⊆ frozen adj(x). Do not run LDLᵀ, do not
        // count for pMax / first-hit. adj_x is nullptr on the skeleton path.
        // 1D: adj_s0=0 so every x shares one length-p row. 2D: adj[x, s].
        if (adj_x != nullptr) {
            bool subset = true;
            for (int j = 0; j < k; ++j) {
                if (adj_x[static_cast<int64_t>(x) * adj_s0 +
                          static_cast<int64_t>(Sloc[j]) * adj_s1] == 0) {
                    subset = false;
                    break;
                }
            }
            if (subset) {
                if (r_out != nullptr) {
                    r_out[b] = static_cast<T>(NAN);
                }
                return;
            }
        }
    }
    T r = fused_r<T, kSMax>::eval(C, cs0, cs1, x, y, Sloc, k, eps);
    if (min_abs != nullptr) {
        atomic_min_abs(min_abs + e, static_cast<double>(r));
    }
    if (r_out != nullptr) {
        r_out[b] = r;
    }
    if (!r_is_indep(static_cast<double>(r), n_samples, k, tau, ge_alpha)) {
        return;
    }
    if (write_g) {
        G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] = 0;
        G[static_cast<int64_t>(y) * gs0 + static_cast<int64_t>(x) * gs1] = 0;
    }
    // PC-stable union semantics: every independent S of the depth contributes
    // its members to the pair's separating-set union (one record per pair,
    // merged across both directed enumerations). Plain byte stores; races
    // only ever write the same 1.
    if (union_marks != nullptr) {
        const int64_t row = rows != nullptr ? rows[e] : e;
        if (hit_flags != nullptr) {
            hit_flags[row] = 1;
        }
        for (int j = 0; j < k; ++j) {
            union_marks[row * u_stride + Sloc[j]] = 1;
        }
    }
    atomic_min_i64(&first_rank[e], b - offsets[e]);
}

// ===========================================================================
// Small-k pair-major CI kernels (PC fast path, k = 0..3).
// One CUDA block per live directed pair. See ci_pair_major_kernel for the
// layout; ci_pair_major_k0_kernel collapses k = 0 to one thread per pair.
// ===========================================================================

// Pair-major CI enumeration for the small PC depths (k <= 3). One CUDA block
// per live directed pair ``e``: ``blockIdx`` indexes the pair directly, so the
// per-test binary search over ``offsets`` (``pair_index``) disappears and the
// per-pair ``sorted_find_idx`` for ``y`` is paid once per block instead of once
// per test. Threads stride over the pair's local combinadic ranks; the rank
// ordering is identical to ``ci_from_comb_kernel`` (``b - offsets[e]``), so
// first-hit / union bookkeeping is unchanged.
//
// Contract: ``ncomb[e] == C(deg(px[e]) - 1, k)``, ``first_rank`` is
// pre-filled with RANK_MAX, and the local rank base is zero (the PC depth
// driver always passes rank_base = 0). ``union_marks`` (uint8 [rows, p]) and
// ``hit_flags`` (uint8 [rows]) are only touched on an independent test;
// byte stores race only with identical 1s.
template <typename T, int kSMax>
__global__ void ci_pair_major_kernel(
    const T* __restrict__ C, long long cs0, long long cs1, int8_t* __restrict__ G, int64_t gs0,
    int64_t gs1, const int* __restrict__ nbrs, const int* __restrict__ deg,
    const int* __restrict__ px, const int* __restrict__ py, const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ ncomb, int64_t* __restrict__ first_rank, int64_t E, int max_d,
    int k, int n_samples, double tau, int ge_alpha, int write_g, T eps, T* __restrict__ r_out,
    const int8_t* __restrict__ adj_x, int64_t adj_s0, int64_t adj_s1,
    uint8_t* __restrict__ union_marks, const int* __restrict__ rows, int64_t u_stride,
    uint8_t* __restrict__ hit_flags, unsigned long long* __restrict__ min_abs) {
    // 2D grid: blockIdx.x + blockIdx.y * gridDim.x == e (gridDim.x is capped
    // by the launcher so E can exceed the 1D grid limit).
    int64_t e = static_cast<int64_t>(blockIdx.x) + static_cast<int64_t>(blockIdx.y) * gridDim.x;
    if (e >= E) {
        return;
    }
    const int x = px[e];
    const int y = py[e];
    if (write_g && G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] == 0 &&
        r_out == nullptr) {
        return;
    }
    const int64_t c = ncomb[e];
    if (c <= 0) {
        return;
    }
    const int d = deg[x];
    const int* row = nbrs + static_cast<int64_t>(x) * max_d;
    int y_idx = -1;
    if (k > 0) {
        y_idx = sorted_find_idx(row, d, y);
    }
    const int64_t uid = rows != nullptr ? rows[e] : e;
    uint8_t* urow = union_marks != nullptr ? union_marks + uid * u_stride : nullptr;
    const int8_t* arow = adj_x != nullptr ? adj_x + static_cast<int64_t>(x) * adj_s0 : nullptr;
    for (int64_t local = threadIdx.x; local < c; local += static_cast<int64_t>(blockDim.x)) {
        int Sloc[kSMax];
        if (k > 0) {
            if (k == 1) {
                int pos = static_cast<int>(local);
                if (y_idx >= 0 && pos >= y_idx) {
                    ++pos;
                }
                Sloc[0] = row[pos];
            } else {
                int n = d - (y_idx >= 0 ? 1 : 0);
                int combo[kSMax];
                unrank_lex(n, k, local, combo);
                for (int j = 0; j < k; ++j) {
                    int pos = combo[j];
                    if (y_idx >= 0 && pos >= y_idx) {
                        ++pos;
                    }
                    Sloc[j] = row[pos];
                }
            }
            if (arow != nullptr) {
                bool subset = true;
                for (int j = 0; j < k; ++j) {
                    if (arow[static_cast<int64_t>(Sloc[j]) * adj_s1] == 0) {
                        subset = false;
                        break;
                    }
                }
                if (subset) {
                    if (r_out != nullptr) {
                        r_out[offsets[e] + local] = static_cast<T>(NAN);
                    }
                    continue;
                }
            }
        }
        T r = fused_r<T, kSMax>::eval(C, cs0, cs1, x, y, Sloc, k, eps);
        if (min_abs != nullptr) {
            atomic_min_abs(min_abs + e, static_cast<double>(r));
        }
        if (r_out != nullptr) {
            r_out[offsets[e] + local] = r;
        }
        if (!r_is_indep(static_cast<double>(r), n_samples, k, tau, ge_alpha)) {
            continue;
        }
        if (write_g) {
            G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] = 0;
            G[static_cast<int64_t>(y) * gs0 + static_cast<int64_t>(x) * gs1] = 0;
        }
        if (urow != nullptr) {
            if (hit_flags != nullptr) {
                hit_flags[uid] = 1;
            }
            for (int j = 0; j < k; ++j) {
                urow[Sloc[j]] = 1;
            }
        }
        atomic_min_i64(&first_rank[e], local);
    }
}

// k = 0 is exactly one test per live directed pair: one flat thread per pair,
// no offsets / combinadic work at all. The pair set has already been reduced
// to x < y by pc_live_pairs_kernel when the symmetric driver is active, so
// this kernel stays direction-agnostic; claim_sepsets_kernel(..., mirror=1)
// then writes the empty sepset into both ranges cells.
__global__ void pmax_update_kernel(const unsigned long long* __restrict__ min_abs,
                                   const int* __restrict__ px, const int* __restrict__ py,
                                   int64_t E, double* __restrict__ pmax, int p, int n_samples,
                                   int k, int symmetric) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }
    const unsigned long long bits = min_abs[e];
    if (bits == ~0ull) {
        return;
    }
    const int n_dof = n_samples - k - 3;
    if (n_dof <= 0) {
        return;
    }
    const double r = __longlong_as_double(bits);
    const double dof = static_cast<double>(n_dof);
    const double z = sqrt(dof) * atanh(r);
    const double pv = erfc(z * 0.7071067811865476);
    const int x = px[e];
    const int y = py[e];
    double* xy = pmax + static_cast<int64_t>(x) * p + y;
    *xy = fmax(*xy, pv);
    if (symmetric) {
        double* yx = pmax + static_cast<int64_t>(y) * p + x;
        *yx = fmax(*yx, pv);
    }
}

template <typename T>
__global__ void ci_pair_major_k0_kernel(
    const T* __restrict__ C, long long cs0, long long cs1, int8_t* __restrict__ G, int64_t gs0,
    int64_t gs1, const int* __restrict__ px, const int* __restrict__ py,
    int64_t* __restrict__ first_rank, int64_t E, int n_samples, double tau, int ge_alpha,
    int write_g, T eps, T* __restrict__ r_out, const int64_t* __restrict__ offsets,
    uint8_t* __restrict__ union_marks, const int* __restrict__ rows, int64_t u_stride,
    uint8_t* __restrict__ hit_flags, unsigned long long* __restrict__ min_abs) {
    int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }
    const int x = px[e];
    const int y = py[e];
    if (write_g && G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] == 0 &&
        r_out == nullptr) {
        return;
    }
    T r = fused_r<T, 4>::eval(C, cs0, cs1, x, y, nullptr, 0, eps);
    if (min_abs != nullptr) {
        atomic_min_abs(min_abs + e, static_cast<double>(r));
    }
    if (r_out != nullptr) {
        r_out[offsets[e]] = r;
    }
    if (!r_is_indep(static_cast<double>(r), n_samples, 0, tau, ge_alpha)) {
        return;
    }
    if (write_g) {
        G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] = 0;
        G[static_cast<int64_t>(y) * gs0 + static_cast<int64_t>(x) * gs1] = 0;
    }
    if (union_marks != nullptr) {
        const int64_t uid = rows != nullptr ? rows[e] : e;
        if (hit_flags != nullptr) {
            hit_flags[uid] = 1;
        }
        (void)u_stride;
    }
    atomic_min_i64(&first_rank[e], 0);
}

__device__ bool sorted_contains(const int* row, int d, int v) {
    int lo = 0;
    int hi = d;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int mv = row[mid];
        if (mv == v) {
            return true;
        }
        if (mv < v) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return false;
}

__device__ bool sorted_subset(const int* row, int d, const int* S, int k) {
    int i = 0;
    for (int j = 0; j < k; ++j) {
        int need = S[j];
        while (i < d && row[i] < need) {
            ++i;
        }
        if (i >= d || row[i] != need) {
            return false;
        }
        ++i;
    }
    return true;
}

__device__ int64_t comb_rank_lex(const int* S, int k, int n) {
    int64_t r = 0;
    int start = 0;
    for (int j = 0; j < k; ++j) {
        int chosen = S[j];
        int remaining = k - j - 1;
        for (int x = start; x < chosen; ++x) {
            int64_t c = binom_sat(n - x - 1, remaining);
            if (c < 0) {
                return -1;
            }
            r += c;
        }
        start = chosen + 1;
    }
    return r;
}

__global__ void mark_s_k1_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg, int p,
                                 int max_d, uint8_t* __restrict__ mark) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= p) {
        return;
    }
    int d = deg[x];
    if (d < 2) {
        return;
    }
    const int* row = nbrs + static_cast<int64_t>(x) * max_d;
    for (int a = 0; a < d; ++a) {
        mark[row[a]] = 1;
    }
}

__global__ void mark_s_k2_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg, int p,
                                 int max_d, uint8_t* __restrict__ mark) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= p) {
        return;
    }
    int d = deg[x];
    const int* row = nbrs + static_cast<int64_t>(x) * max_d;
    for (int a = 0; a < d; ++a) {
        int s0 = row[a];
        for (int b = a + 1; b < d; ++b) {
            mark[static_cast<int64_t>(s0) * p + row[b]] = 1;
        }
    }
}

__global__ void mark_s_ranks_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg,
                                    int p, int max_d, int k, uint8_t* __restrict__ bits,
                                    int64_t ncomb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= p) {
        return;
    }
    int d = deg[x];
    int64_t nloc = binom_sat(d, k);
    if (nloc <= 0) {
        return;
    }
    const int* row = nbrs + static_cast<int64_t>(x) * max_d;
    int combo[kMaxCond];
    int S[kMaxCond];
    for (int64_t local = 0; local < nloc; ++local) {
        unrank_lex(d, k, local, combo);
        for (int j = 0; j < k; ++j) {
            S[j] = row[combo[j]];
        }
        int64_t r = comb_rank_lex(S, k, p);
        if (r >= 0 && r < ncomb) {
            bits[r] = 1;
        }
    }
}

__global__ void unrank_rows_kernel(const int64_t* __restrict__ idx, int64_t U, int p, int k,
                                   int* __restrict__ S) {
    int64_t u = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (u >= U) {
        return;
    }
    int combo[kMaxCond];
    unrank_lex(p, k, idx[u], combo);
    for (int j = 0; j < k; ++j) {
        S[u * k + j] = combo[j];
    }
}

// ===========================================================================
// Possible-D-SEP (PDSEP) helpers. Unlike the depth kernels, these enumerate
// S-major: for one conditioning set S, find every (x, y) with x adjacent to
// all of S, y adjacent to x, y not in S, and the edge present. With
// ``undirected`` the pair is emitted once: when x > y and S is also contained
// in nbrs(y), the higher-endpoint hub will emit it, so this direction is
// skipped; otherwise it is emitted as (y, x). Used by FCI's
// possible-D-SEP driver, not by the PC skeleton.
// ===========================================================================

__device__ int64_t each_xy_for_S(const int8_t* G, int64_t gs0, int64_t gs1, const int* nbrs,
                                 const int* deg, int max_d, const int* S, int k, int undirected,
                                 int* x_out, int* y_out, int* Sout, int64_t out_off, int write) {
    int64_t nwrite = 0;
    int s0 = S[0];
    int d0 = deg[s0];
    const int* row0 = nbrs + static_cast<int64_t>(s0) * max_d;
    for (int t = 0; t < d0; ++t) {
        int x = row0[t];
        bool ok = true;
        for (int j = 1; j < k && ok; ++j) {
            ok = sorted_contains(nbrs + static_cast<int64_t>(S[j]) * max_d, deg[S[j]], x);
        }
        if (!ok) {
            continue;
        }
        int dx = deg[x];
        const int* nx = nbrs + static_cast<int64_t>(x) * max_d;
        for (int yi = 0; yi < dx; ++yi) {
            int y = nx[yi];
            if (sorted_contains(S, k, y)) {
                continue;
            }
            if (G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] == 0) {
                continue;
            }
            int xo = x;
            int yo = y;
            if (undirected && x > y) {
                if (sorted_subset(nbrs + static_cast<int64_t>(y) * max_d, deg[y], S, k)) {
                    continue;
                }
                xo = y;
                yo = x;
            }
            if (write) {
                int64_t b = out_off + nwrite;
                x_out[b] = xo;
                y_out[b] = yo;
                for (int j = 0; j < k; ++j) {
                    Sout[b * k + j] = S[j];
                }
            }
            ++nwrite;
        }
    }
    return nwrite;
}

__global__ void count_s_tests_kernel(const int8_t* __restrict__ G, int64_t gs0, int64_t gs1,
                                     const int* __restrict__ nbrs, const int* __restrict__ deg,
                                     const int* __restrict__ unique_S, int64_t U, int max_d, int k,
                                     int undirected, int64_t* __restrict__ counts) {
    int64_t u = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (u >= U) {
        return;
    }
    counts[u] = each_xy_for_S(G, gs0, gs1, nbrs, deg, max_d, unique_S + u * k, k, undirected,
                              nullptr, nullptr, nullptr, 0, 0);
}

__global__ void fill_s_tests_kernel(const int8_t* __restrict__ G, int64_t gs0, int64_t gs1,
                                    const int* __restrict__ nbrs, const int* __restrict__ deg,
                                    const int* __restrict__ unique_S,
                                    const int64_t* __restrict__ offsets, int64_t U, int max_d,
                                    int k, int undirected, int* __restrict__ x_out,
                                    int* __restrict__ y_out, int* __restrict__ Sout) {
    int64_t u = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (u >= U) {
        return;
    }
    each_xy_for_S(G, gs0, gs1, nbrs, deg, max_d, unique_S + u * k, k, undirected, x_out, y_out,
                  Sout, offsets[u], 1);
}

__global__ void count_nbr_combos_kernel(const int* __restrict__ deg, int p, int k,
                                        int64_t* __restrict__ counts) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= p) {
        return;
    }
    counts[x] = binom_sat(deg[x], k);
}

__global__ void fill_nbr_combos_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg,
                                       const int64_t* __restrict__ offsets, int p, int max_d, int k,
                                       int* __restrict__ S) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= p) {
        return;
    }
    int64_t nloc = offsets[x + 1] - offsets[x];
    if (nloc <= 0) {
        return;
    }
    int d = deg[x];
    const int* row = nbrs + static_cast<int64_t>(x) * max_d;
    int combo[kMaxCond];
    int64_t base = offsets[x];
    for (int64_t local = 0; local < nloc; ++local) {
        unrank_lex(d, k, local, combo);
        for (int j = 0; j < k; ++j) {
            S[(base + local) * k + j] = row[combo[j]];
        }
    }
}

template <typename T>
__global__ void first_hit_kernel(const T* __restrict__ r, const int64_t* __restrict__ offsets,
                                 int64_t* __restrict__ first_rank, int64_t B, int64_t E,
                                 int n_samples, int k, double tau, int ge_alpha) {
    int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }
    if (!r_is_indep(static_cast<double>(r[b]), n_samples, k, tau, ge_alpha)) {
        return;
    }
    int64_t e = pair_index(offsets, E, b);
    atomic_min_i64(&first_rank[e], b - offsets[e]);
}

__global__ void fill_at_ranks_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg,
                                     const int* __restrict__ px, const int* __restrict__ py,
                                     const int64_t* __restrict__ ranks, int64_t E, int max_d, int k,
                                     int* __restrict__ x_out, int* __restrict__ y_out,
                                     int* __restrict__ S) {
    int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }
    int x = px[e];
    int y = py[e];
    x_out[e] = x;
    y_out[e] = y;
    if (k <= 0) {
        return;
    }
    int d = deg[x];
    const int* row = nbrs + static_cast<int64_t>(x) * max_d;
    int y_idx = sorted_find_idx(row, d, y);
    int n = d - (y_idx >= 0 ? 1 : 0);
    int combo[kMaxCond];
    unrank_lex(n, k, ranks[e], combo);
    for (int j = 0; j < k; ++j) {
        int pos = combo[j];
        if (y_idx >= 0 && pos >= y_idx) {
            ++pos;
        }
        S[e * k + j] = row[pos];
    }
}

template <typename T>
__global__ void hits_from_r_kernel(const T* __restrict__ r, uint8_t* __restrict__ hit, int64_t B,
                                   int n_samples, int k, double tau, int ge_alpha) {
    int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }
    hit[b] = r_is_indep(static_cast<double>(r[b]), n_samples, k, tau, ge_alpha) ? 1 : 0;
}

__global__ void delete_if_indep_kernel(int8_t* __restrict__ G, int64_t s0, int64_t s1,
                                       const int* __restrict__ x, const int* __restrict__ y,
                                       const uint8_t* __restrict__ indep, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int xi = x[i];
    int yi = y[i];
    if (G[xi * s0 + yi * s1] == 0) {
        return;
    }
    if (!indep[i]) {
        return;
    }
    G[xi * s0 + yi * s1] = 0;
    G[yi * s0 + xi * s1] = 0;
}

__global__ void edge_present_kernel(const int8_t* __restrict__ G, int64_t s0, int64_t s1,
                                    const int* __restrict__ x, const int* __restrict__ y,
                                    uint8_t* __restrict__ present, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int xi = x[i];
    int yi = y[i];
    present[i] = (G[xi * s0 + yi * s1] != 0) ? 1 : 0;
}

__global__ void apply_hits_kernel(int8_t* __restrict__ G, int64_t s0, int64_t s1,
                                  const int* __restrict__ x, const int* __restrict__ y, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int xi = x[i];
    int yi = y[i];
    G[xi * s0 + yi * s1] = 0;
    G[yi * s0 + xi * s1] = 0;
}

__global__ void pack_px_lt_py_kernel(const int* __restrict__ px, const int* __restrict__ py,
                                     int64_t n, int* __restrict__ px_out, int* __restrict__ py_out,
                                     int64_t* __restrict__ ranks_out,
                                     int64_t* __restrict__ ncomb_out, int* __restrict__ E_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (px[i] < py[i]) {
        int d = atomicAdd(E_out, 1);
        px_out[d] = px[i];
        py_out[d] = py[i];
        ranks_out[d] = 0;
        ncomb_out[d] = 1;
    }
}

__global__ void pack_ncomb_pos_kernel(const int* __restrict__ px, const int* __restrict__ py,
                                      const int64_t* __restrict__ ncomb, int64_t n,
                                      int* __restrict__ px_out, int* __restrict__ py_out,
                                      int64_t* __restrict__ ranks_out,
                                      int64_t* __restrict__ ncomb_out, int* __restrict__ E_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (ncomb[i] > 0) {
        int d = atomicAdd(E_out, 1);
        px_out[d] = px[i];
        py_out[d] = py[i];
        ranks_out[d] = 0;
        ncomb_out[d] = ncomb[i];
    }
}

__global__ void compact_live_kernel(const int* __restrict__ px, const int* __restrict__ py,
                                    const int64_t* __restrict__ ranks,
                                    const int64_t* __restrict__ ncomb,
                                    const uint8_t* __restrict__ hit, const int* __restrict__ S,
                                    int k, int64_t E, int* __restrict__ px_out,
                                    int* __restrict__ py_out, int64_t* __restrict__ ranks_out,
                                    int64_t* __restrict__ ncomb_out, int* __restrict__ hit_x,
                                    int* __restrict__ hit_y, int* __restrict__ hit_S,
                                    int* __restrict__ E_out, int* __restrict__ H_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= E) {
        return;
    }
    if (hit[i]) {
        int d = atomicAdd(H_out, 1);
        hit_x[d] = px[i];
        hit_y[d] = py[i];
        if (k > 0 && S != nullptr) {
            const int* src = S + i * static_cast<int64_t>(k);
            int* dst = hit_S + static_cast<int64_t>(d) * k;
            for (int j = 0; j < k; ++j) {
                dst[j] = src[j];
            }
        }
    }
    if (!hit[i] && ranks[i] + 1 < ncomb[i]) {
        int d = atomicAdd(E_out, 1);
        px_out[d] = px[i];
        py_out[d] = py[i];
        ncomb_out[d] = ncomb[i];
        ranks_out[d] = ranks[i] + 1;
    }
}

// ===========================================================================
// Sepset CSR bookkeeping.
//   claim_sepsets_kernel        first-hit claim (FCI / possible-D-SEP)
//   zero_claimed_kernel         k = 0 deletion follow-up for the flat path
//   commit_union_mark_kernel    canonicalize hit pairs into a p*p table
//   commit_union_write[_fused]  merge the <=2 directions, append S, write ranges
//   pdsep_commit_walk_kernel    sequential last-writer commit for possible-D-SEP
// Layout: ranges[p, p, 2] = (start, end) into the 1-D int64 S; (-1,-1) means
// the edge is still present, (0,0) an empty sepset (k = 0).
// ===========================================================================

// One thread per pair. First hit claims ``ranges[x,y]`` via CAS on start.
// ``ranges[x,y] = (start, end)`` into 1D ``S``. Still an edge: ``(-1,-1)``.
// Empty sepset (k=0): ``(0,0)``. ``(x,y)`` and ``(y,x)`` claim separately.
// ``mirror != 0`` additionally copies the range into ``ranges[y,x]``; the PC
// k = 0 workspace path needs this because it enumerates only x < y. A loser of
// the CAS must not touch ``S``/``nnz`` (the winner owns that record).
// Launch: ceil(E/256) CTAs × 256; int64 atomics + combinadic unrank.
__global__ void claim_sepsets_kernel(
    const int* __restrict__ px, const int* __restrict__ py, const int64_t* __restrict__ first_rank,
    const int64_t* __restrict__ rank_bases, int64_t rank_base, int64_t rank_max,
    const int* __restrict__ nbrs, const int* __restrict__ deg, int64_t E, int p, int max_d, int k,
    int64_t s_cap, int64_t* __restrict__ ranges, int64_t* __restrict__ nnz, int64_t* __restrict__ S,
    int8_t* __restrict__ G, int64_t gs0, int64_t gs1, int mirror, int* __restrict__ overflow) {
    int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E) {
        return;
    }
    int64_t local = first_rank[e];
    if (local >= rank_max) {
        return;
    }
    int x = px[e];
    int y = py[e];
    if (G != nullptr) {
        G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] = 0;
        G[static_cast<int64_t>(y) * gs0 + static_cast<int64_t>(x) * gs1] = 0;
    }
    int64_t* start_cell = ranges + (static_cast<int64_t>(x) * p + y) * 2;
    int64_t old = atomic_cas_i64(start_cell, static_cast<int64_t>(-1), static_cast<int64_t>(-2));
    if (old != static_cast<int64_t>(-1)) {
        return;
    }
    int64_t start = 0;
    int64_t end = 0;
    if (k > 0) {
        start = atomic_add_i64(nnz, static_cast<int64_t>(k));
        end = start + static_cast<int64_t>(k);
        if (end > s_cap) {
            atomicAdd(overflow, 1);
            *start_cell = static_cast<int64_t>(-1);
            return;
        }
        {
            int64_t rank = (rank_bases != nullptr) ? (local + rank_bases[e]) : (local + rank_base);
            int d = deg[x];
            const int* row = nbrs + static_cast<int64_t>(x) * max_d;
            int y_idx = sorted_find_idx(row, d, y);
            int n = d - (y_idx >= 0 ? 1 : 0);
            int combo[kMaxCond];
            unrank_lex(n, k, rank, combo);
            for (int j = 0; j < k; ++j) {
                int pos = combo[j];
                if (y_idx >= 0 && pos >= y_idx) {
                    ++pos;
                }
                S[start + j] = row[pos];
            }
        }
    }
    int64_t* xy = ranges + (static_cast<int64_t>(x) * p + y) * 2;
    xy[1] = end;
    xy[0] = start;
    if (mirror) {
        int64_t* yx = ranges + (static_cast<int64_t>(y) * p + x) * 2;
        yx[0] = start;
        yx[1] = end;
    }
}

// Inclusive prefix sum of the per-pair test counts. One CTA: per-thread
// sequential scan of a contiguous chunk, then a Hillis-Steele scan of the
// chunk sums. ``out[n]`` receives the total (grid must be 1 CTA).
__global__ void scan_counts_i64_kernel(const int64_t* __restrict__ in, int64_t* __restrict__ out,
                                       int n) {
    if (blockIdx.x != 0) {
        return;
    }
    extern __shared__ int64_t sh[];
    const int t = static_cast<int>(threadIdx.x);
    const int nt = static_cast<int>(blockDim.x);
    const int per = (n + nt - 1) / nt;
    const int lo = t * per;
    const int hi = (lo + per < n) ? lo + per : n;
    int64_t s = 0;
    for (int i = lo; i < hi; ++i) {
        s += in[i];
    }
    sh[t] = s;
    __syncthreads();
    for (int off = 1; off < nt; off <<= 1) {
        int64_t v = (t >= off) ? sh[t - off] : 0;
        __syncthreads();
        sh[t] += v;
        __syncthreads();
    }
    const int64_t base = (t == 0) ? 0 : sh[t - 1];
    if (t == 0) {
        out[n] = sh[nt - 1];
    }
    int64_t acc = 0;
    for (int i = lo; i < hi; ++i) {
        out[i] = base + acc;
        acc += in[i];
    }
}

// Zero both triangles of ``G`` for every ordered pair with a claimed CSR
// range (k = 0 empty-sepset deletions). Replaces the host
// ``ranges[:,:,0] >= 0`` mask + masked_fill in the k = 0 depth.
__global__ void zero_claimed_kernel(int8_t* __restrict__ G, int64_t gs0, int64_t gs1,
                                    const int64_t* __restrict__ ranges, int p) {
    int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t nk = static_cast<int64_t>(p) * p;
    if (k >= nk) {
        return;
    }
    if (ranges[k * 2] < 0) {
        return;
    }
    int a = static_cast<int>(k / p);
    int b = static_cast<int>(k % p);
    G[static_cast<int64_t>(a) * gs0 + static_cast<int64_t>(b) * gs1] = 0;
}

// ===========================================================================
// PC-stable depth driver pieces. graph_batch_pc_depth_union (host side, at
// the bottom) glues them together: pc_live_pairs -> per-depth total -> fused
// pair-major CI -> union commit -> G update. The Python counterpart is
// cuda_graph_batch._run_pc_union_depth.
// ===========================================================================

// Live directed pairs of one PC depth in a single launch. One thread per
// ``(x, i)`` slot of the packed neighbor table; a thread owns the directed
// pair ``(x, nbrs[x, i])`` when ``i < deg[x]``. Emits ``px``/``py`` and the
// per-pair test count ``C(deg(x)-1, k)`` (1 at k=0), skipping empty pairs.
// Replaces emit_pairs + count_tests + pack_ncomb for the fused depth.
//
// Symmetry invariant (``symmetric`` != 0, only set for k = 0): emit the pair
// once from the lower endpoint, so a depth costs p*(p-1)/2 tests instead of
// p*(p-1). For x > y the thread returns only when the reverse arc exists in
// G, because the lower endpoint will emit that pair; with an asymmetric G the
// reverse arc may be absent, and the higher endpoint must emit to avoid
// losing the pair. The PC skeleton passes a symmetric G, and the driver
// mirrors the empty sepset into both ranges cells.
__global__ void pc_live_pairs_kernel(const int* __restrict__ nbrs, const int* __restrict__ deg,
                                     int p, int max_d, int k, int* __restrict__ px,
                                     int* __restrict__ py, int64_t* __restrict__ ncomb,
                                     int* __restrict__ E_out, int64_t* __restrict__ first_out,
                                     int symmetric, const int8_t* __restrict__ G, int64_t gs0,
                                     int64_t gs1) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int x = static_cast<int>(idx / max_d);
    if (x >= p) {
        return;
    }
    int i = static_cast<int>(idx % max_d);
    int d = deg[x];
    if (i >= d) {
        return;
    }
    int y = nbrs[static_cast<int64_t>(x) * max_d + i];
    // k = 0 CI is symmetric: enumerate the upper triangle only when the
    // mirrored arc actually exists (G may be asymmetric for direct callers;
    // the PC skeleton always passes a symmetric G). The driver mirrors the
    // empty sepset into both ``ranges`` cells.
    if (symmetric && x > y && G != nullptr &&
        G[static_cast<int64_t>(y) * gs0 + static_cast<int64_t>(x) * gs1] != 0) {
        return;
    }
    int64_t c = (k == 0) ? 1 : binom_sat(d - 1, k);
    if (c <= 0) {
        return;
    }
    int slot = atomicAdd(E_out, 1);
    px[slot] = x;
    py[slot] = y;
    ncomb[slot] = c;
    if (first_out != nullptr) {
        first_out[slot] = std::numeric_limits<int64_t>::max();
    }
}

// Total number of CI tests of the live pairs: ``sum(ncomb[0:E_live])``.
// Single CTA; reads the live count from the device (``E_out``) so the host
// never has to round-trip the pair count before the CI launch.
__global__ void sum_ncomb_live_kernel(const int64_t* __restrict__ ncomb,
                                      const int* __restrict__ E_out, int64_t* __restrict__ total) {
    __shared__ int64_t sh[kBlock];
    const int t = static_cast<int>(threadIdx.x);
    const int n = *E_out;
    int64_t s = 0;
    for (int i = t; i < n; i += static_cast<int>(blockDim.x)) {
        s += ncomb[i];
    }
    sh[t] = s;
    __syncthreads();
    for (int off = 1; off < static_cast<int>(blockDim.x); off <<= 1) {
        int64_t v = (t >= off) ? sh[t - off] : 0;
        __syncthreads();
        sh[t] += v;
        __syncthreads();
    }
    if (t == 0) {
        *total = sh[blockDim.x - 1];
    }
}

// Zero ``G`` for every flagged directed pair (both triangles). Replaces
// hit_flags.nonzero() + index_select + apply_hits in the PC union depth.
__global__ void zero_flagged_kernel(int8_t* __restrict__ G, int64_t gs0, int64_t gs1,
                                    const int* __restrict__ px, const int* __restrict__ py,
                                    const uint8_t* __restrict__ flags, int64_t E) {
    int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E || flags[e] == 0) {
        return;
    }
    int x = px[e];
    int y = py[e];
    G[static_cast<int64_t>(x) * gs0 + static_cast<int64_t>(y) * gs1] = 0;
    G[static_cast<int64_t>(y) * gs0 + static_cast<int64_t>(x) * gs1] = 0;
}

// PC-stable union sepsets, fused on device. Two launches replace the host
// torch chain (canonical key -> argsort -> group -> OR bitmap -> cumsum ->
// scatter) that dominated the skeleton for every depth.
//
// Kernel 1: one thread per flagged directed pair. The pair's canonical key is
// ``min*p + max``; at most two directed rows share a key (``(x,y)`` and
// ``(y,x)``, G is symmetric), so ``tab_a``/``tab_b`` hold them.
// Kernel 2: one thread per canonical key. OR the <=2 member bitmaps, append the
// sorted members to ``S`` at an atomically reserved offset, and write both
// symmetric ``ranges`` cells.
__global__ void commit_union_mark_kernel(const int* __restrict__ px, const int* __restrict__ py,
                                         const uint8_t* __restrict__ flags, int64_t E, int p,
                                         int* __restrict__ tab_a, int* __restrict__ tab_b) {
    int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= E || flags[e] == 0) {
        return;
    }
    int x = px[e];
    int y = py[e];
    int a = x < y ? x : y;
    int b = x < y ? y : x;
    int64_t key = static_cast<int64_t>(a) * p + b;
    int prev = atomicCAS(&tab_a[key], -1, static_cast<int>(e));
    if (prev != -1) {
        atomicExch(&tab_b[key], static_cast<int>(e));
    }
}

__global__ void commit_union_write_kernel(const int* __restrict__ tab_a,
                                          const int* __restrict__ tab_b,
                                          const uint8_t* __restrict__ union_marks, int64_t u_stride,
                                          int p, int64_t* __restrict__ ranges,
                                          int64_t* __restrict__ nnz, int64_t* __restrict__ S,
                                          int64_t s_cap, int* __restrict__ overflow) {
    int64_t key = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t nkey = static_cast<int64_t>(p) * p;
    if (key >= nkey) {
        return;
    }
    int e0 = tab_a[key];
    if (e0 < 0) {
        return;
    }
    int e1 = tab_b[key];
    const uint8_t* r0 = union_marks + static_cast<int64_t>(e0) * u_stride;
    const uint8_t* r1 = e1 >= 0 ? union_marks + static_cast<int64_t>(e1) * u_stride : nullptr;
    int cnt = 0;
    for (int v = 0; v < p; ++v) {
        if (r0[v] != 0 || (r1 != nullptr && r1[v] != 0)) {
            ++cnt;
        }
    }
    int a = static_cast<int>(key / p);
    int b = static_cast<int>(key % p);
    int64_t* xy = ranges + (static_cast<int64_t>(a) * p + b) * 2;
    int64_t* yx = ranges + (static_cast<int64_t>(b) * p + a) * 2;
    if (cnt == 0) {
        xy[0] = 0;
        xy[1] = 0;
        yx[0] = 0;
        yx[1] = 0;
        return;
    }
    int64_t start = atomic_add_i64(nnz, static_cast<int64_t>(cnt));
    int64_t end = start + static_cast<int64_t>(cnt);
    if (end > s_cap) {
        atomicAdd(overflow, 1);
        return;
    }
    int64_t pos = start;
    for (int v = 0; v < p; ++v) {
        if (r0[v] != 0 || (r1 != nullptr && r1[v] != 0)) {
            S[pos++] = static_cast<int64_t>(v);
        }
    }
    xy[0] = start;
    xy[1] = end;
    yx[0] = start;
    yx[1] = end;
}

// Fused commit: kernel 2 above plus the G update of ``zero_flagged_kernel``.
// The mark tables are workspace-cached, so this kernel also resets the two
// entries it consumed: the next depth starts with an all -1 table without a
// full ``p*p`` fill. Every key has a unique owner (thread ``key``), so the
// reset cannot race: only the thread that read (-1-free) entries writes them
// back. ``px``/``py`` identify the hit directed pairs for the G stores.
//
// Why one table entry per canonical key is enough: PC-stable hits at most the
// two directions (x,y) and (y,x) of a pair in a depth, and the query kernel
// stores them in tab_a/tab_b keyed by ``min*p + max``.
__global__ void commit_union_write_fused_kernel(
    int* __restrict__ tab_a, int* __restrict__ tab_b, const uint8_t* __restrict__ union_marks,
    int64_t u_stride, int p, int64_t* __restrict__ ranges, int64_t* __restrict__ nnz,
    int64_t* __restrict__ S, int64_t s_cap, int* __restrict__ overflow, const int* __restrict__ px,
    const int* __restrict__ py, int8_t* __restrict__ G, int64_t gs0, int64_t gs1) {
    int64_t key = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t nkey = static_cast<int64_t>(p) * p;
    if (key >= nkey) {
        return;
    }
    int e0 = tab_a[key];
    if (e0 < 0) {
        return;
    }
    int e1 = tab_b[key];
    tab_a[key] = -1;
    tab_b[key] = -1;
    if (G != nullptr) {
        int x0 = px[e0];
        int y0 = py[e0];
        G[static_cast<int64_t>(x0) * gs0 + static_cast<int64_t>(y0) * gs1] = 0;
        G[static_cast<int64_t>(y0) * gs0 + static_cast<int64_t>(x0) * gs1] = 0;
        if (e1 >= 0) {
            int x1 = px[e1];
            int y1 = py[e1];
            G[static_cast<int64_t>(x1) * gs0 + static_cast<int64_t>(y1) * gs1] = 0;
            G[static_cast<int64_t>(y1) * gs0 + static_cast<int64_t>(x1) * gs1] = 0;
        }
    }
    const uint8_t* r0 = union_marks + static_cast<int64_t>(e0) * u_stride;
    const uint8_t* r1 = e1 >= 0 ? union_marks + static_cast<int64_t>(e1) * u_stride : nullptr;
    int cnt = 0;
    for (int v = 0; v < p; ++v) {
        if (r0[v] != 0 || (r1 != nullptr && r1[v] != 0)) {
            ++cnt;
        }
    }
    int a = static_cast<int>(key / p);
    int b = static_cast<int>(key % p);
    int64_t* xy = ranges + (static_cast<int64_t>(a) * p + b) * 2;
    int64_t* yx = ranges + (static_cast<int64_t>(b) * p + a) * 2;
    if (cnt == 0) {
        xy[0] = 0;
        xy[1] = 0;
        yx[0] = 0;
        yx[1] = 0;
        return;
    }
    int64_t start = atomic_add_i64(nnz, static_cast<int64_t>(cnt));
    int64_t end = start + static_cast<int64_t>(cnt);
    if (end > s_cap) {
        atomicAdd(overflow, 1);
        return;
    }
    int64_t pos = start;
    for (int v = 0; v < p; ++v) {
        if (r0[v] != 0 || (r1 != nullptr && r1[v] != 0)) {
            S[pos++] = static_cast<int64_t>(v);
        }
    }
    xy[0] = start;
    xy[1] = end;
    yx[0] = start;
    yx[1] = end;
}

// overwrite both triangles (sequential pdsep_v2 last-writer). Each thread
// grid-strides the directed pairs of the current x. Unrank S from Pdsep(x).
__global__ void pdsep_commit_walk_kernel(const int* __restrict__ py, const int* __restrict__ hit_k,
                                         const int64_t* __restrict__ hit_rank,
                                         const int64_t* __restrict__ row_off,
                                         const int* __restrict__ nbrs, const int* __restrict__ deg,
                                         int p, int max_d, int64_t s_cap,
                                         int64_t* __restrict__ ranges, int64_t* __restrict__ nnz,
                                         int64_t* __restrict__ S, int* __restrict__ overflow) {
    if (blockIdx.x != 0) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int nt = static_cast<int>(blockDim.x);
    for (int x = 0; x < p; ++x) {
        const int64_t e0 = row_off[x];
        const int64_t e1 = row_off[x + 1];
        for (int64_t e = e0 + tid; e < e1; e += nt) {
            const int k = hit_k[e];
            if (k < 1 || k > kMaxCond) {
                continue;
            }
            const int y = py[e];
            int64_t start = 0;
            int64_t end = 0;
            start = atomic_add_i64(nnz, static_cast<int64_t>(k));
            end = start + static_cast<int64_t>(k);
            if (end > s_cap) {
                atomicAdd(overflow, 1);
                continue;
            }
            {
                const int64_t rank = hit_rank[e];
                const int d = deg[x];
                const int* row = nbrs + static_cast<int64_t>(x) * max_d;
                const int y_idx = sorted_find_idx(row, d, y);
                const int n = d - (y_idx >= 0 ? 1 : 0);
                int combo[kMaxCond];
                unrank_lex(n, k, rank, combo);
                for (int j = 0; j < k; ++j) {
                    int pos = combo[j];
                    if (y_idx >= 0 && pos >= y_idx) {
                        ++pos;
                    }
                    S[start + j] = row[pos];
                }
            }
            int64_t* xy = ranges + (static_cast<int64_t>(x) * p + y) * 2;
            int64_t* yx = ranges + (static_cast<int64_t>(y) * p + x) * 2;
            xy[1] = end;
            xy[0] = start;
            yx[1] = end;
            yx[0] = start;
        }
        __syncthreads();
    }
}

int grid(int64_t n) {
    int64_t blocks = (n + kBlock - 1) / kBlock;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        blocks = std::numeric_limits<int>::max();
    }
    return static_cast<int>(blocks);
}

const int8_t* g_ptr(const torch::Tensor& G) { return G.data_ptr<int8_t>(); }

void touch_graph_batch_anon() {
    touch_cuda_kernel(degrees_kernel);
    touch_cuda_kernel(compact_kernel);
    touch_cuda_kernel(emit_pairs_kernel);
    touch_cuda_kernel(count_tests_kernel);
    touch_cuda_kernel(fill_batch_kernel);
    touch_cuda_kernel(mark_s_k1_kernel);
    touch_cuda_kernel(mark_s_k2_kernel);
    touch_cuda_kernel(mark_s_ranks_kernel);
    touch_cuda_kernel(unrank_rows_kernel);
    touch_cuda_kernel(count_s_tests_kernel);
    touch_cuda_kernel(fill_s_tests_kernel);
    touch_cuda_kernel(count_nbr_combos_kernel);
    touch_cuda_kernel(fill_nbr_combos_kernel);
    touch_cuda_kernel(fill_at_ranks_kernel);
    touch_cuda_kernel(delete_if_indep_kernel);
    touch_cuda_kernel(edge_present_kernel);
    touch_cuda_kernel(apply_hits_kernel);
    touch_cuda_kernel(pack_px_lt_py_kernel);
    touch_cuda_kernel(pack_ncomb_pos_kernel);
    touch_cuda_kernel(compact_live_kernel);
    touch_cuda_kernel(claim_sepsets_kernel);
    touch_cuda_kernel(pdsep_commit_walk_kernel);
#if WHYDRA_ENABLE_FLOAT32
    touch_cuda_kernel(ci_from_comb_kernel<float, 4>);
    touch_cuda_kernel(ci_from_comb_kernel<float, kMaxFused>);
#endif
    touch_cuda_kernel(ci_from_comb_kernel<double, 4>);
    touch_cuda_kernel(ci_from_comb_kernel<double, kMaxFused>);
#if WHYDRA_ENABLE_FLOAT32
    touch_cuda_kernel(first_hit_kernel<float>);
#endif
    touch_cuda_kernel(first_hit_kernel<double>);
#if WHYDRA_ENABLE_FLOAT32
    touch_cuda_kernel(hits_from_r_kernel<float>);
#endif
    touch_cuda_kernel(hits_from_r_kernel<double>);
}

} // namespace

// ===========================================================================
// Host-side torch::Tensor bindings. Each wrapper validates dtypes/shapes,
// resolves the current CUDA stream and launches the corresponding kernel.
// The Python wrappers in cuda_graph_batch.py mirror these signatures; the
// PYBIND11_MODULE at the bottom is the public ABI (do not rename these).
// ===========================================================================

void graph_batch_degrees(torch::Tensor G, torch::Tensor deg) {
    TORCH_CHECK(G.is_cuda() && deg.is_cuda(), "degrees: CUDA tensors");
    TORCH_CHECK(G.dim() == 2 && G.size(0) == G.size(1), "G must be square");
    TORCH_CHECK(G.scalar_type() == at::kChar, "G must be int8");
    TORCH_CHECK(deg.scalar_type() == at::kInt && deg.is_contiguous(), "deg int32 contiguous");
    const int p = static_cast<int>(G.size(0));
    TORCH_CHECK(deg.numel() == p, "deg length");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    degrees_kernel<<<grid(p), kBlock, 0, stream>>>(g_ptr(G), G.stride(0), G.stride(1), p,
                                                   deg.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_compact(torch::Tensor G, torch::Tensor nbrs, torch::Tensor deg) {
    TORCH_CHECK(G.is_cuda() && nbrs.is_cuda(), "compact: CUDA tensors");
    TORCH_CHECK(G.scalar_type() == at::kChar, "G must be int8");
    TORCH_CHECK(nbrs.scalar_type() == at::kInt && nbrs.is_contiguous(), "nbrs int32");
    TORCH_CHECK(nbrs.dim() == 2 && nbrs.size(0) == G.size(0), "nbrs (p, max_d)");
    const int p = static_cast<int>(G.size(0));
    const int max_d = static_cast<int>(nbrs.size(1));
    (void)deg;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    compact_kernel<<<grid(p), kBlock, 0, stream>>>(g_ptr(G), G.stride(0), G.stride(1), p, max_d,
                                                   nbrs.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_emit_pairs(torch::Tensor nbrs, torch::Tensor deg, torch::Tensor row_off,
                            torch::Tensor px, torch::Tensor py) {
    TORCH_CHECK(nbrs.is_cuda() && deg.is_cuda() && row_off.is_cuda(), "emit_pairs CUDA");
    TORCH_CHECK(px.is_contiguous() && py.is_contiguous(), "px/py contiguous");
    const int p = static_cast<int>(deg.numel());
    const int max_d = static_cast<int>(nbrs.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    emit_pairs_kernel<<<grid(p), kBlock, 0, stream>>>(nbrs.data_ptr<int>(), deg.data_ptr<int>(),
                                                      row_off.data_ptr<int64_t>(), p, max_d,
                                                      px.data_ptr<int>(), py.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_count_tests(torch::Tensor nbrs, torch::Tensor deg, torch::Tensor px,
                             torch::Tensor py, int64_t k, torch::Tensor counts) {
    TORCH_CHECK(nbrs.is_cuda() && px.is_cuda() && counts.is_cuda(), "count_tests CUDA");
    TORCH_CHECK(k >= 0 && k <= kMaxCond, "k out of range");
    const int64_t E = px.numel();
    const int max_d = static_cast<int>(nbrs.size(1));
    if (E == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    count_tests_kernel<<<grid(E), kBlock, 0, stream>>>(
        nbrs.data_ptr<int>(), deg.data_ptr<int>(), px.data_ptr<int>(), py.data_ptr<int>(), E, max_d,
        static_cast<int>(k), counts.data_ptr<int64_t>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_fill(torch::Tensor nbrs, torch::Tensor deg, torch::Tensor px, torch::Tensor py,
                      torch::Tensor offsets, int64_t k, int64_t rank_base, torch::Tensor rank_bases,
                      torch::Tensor x, torch::Tensor y, torch::Tensor S) {
    TORCH_CHECK(nbrs.is_cuda() && x.is_cuda() && S.is_cuda(), "fill CUDA");
    TORCH_CHECK(k >= 0 && k <= kMaxCond, "k out of range");
    const int64_t B = x.numel();
    const int64_t E = px.numel();
    if (B == 0) {
        return;
    }
    TORCH_CHECK(offsets.numel() == E + 1, "offsets length E+1");
    TORCH_CHECK(S.size(0) == B && S.size(1) == k, "S (B, k)");
    const int max_d = static_cast<int>(nbrs.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int* Sptr = k > 0 ? S.data_ptr<int>() : nullptr;
    const int64_t* rbp =
        rank_bases.defined() && rank_bases.numel() == E ? rank_bases.data_ptr<int64_t>() : nullptr;
    fill_batch_kernel<<<grid(B), kBlock, 0, stream>>>(
        nbrs.data_ptr<int>(), deg.data_ptr<int>(), px.data_ptr<int>(), py.data_ptr<int>(),
        offsets.data_ptr<int64_t>(), rbp, E, B, max_d, static_cast<int>(k), rank_base,
        x.data_ptr<int>(), y.data_ptr<int>(), Sptr);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_ci_from_comb(
    torch::Tensor C, torch::Tensor G, torch::Tensor nbrs, torch::Tensor deg, torch::Tensor px,
    torch::Tensor py, torch::Tensor offsets, int64_t k, int64_t rank_base, torch::Tensor rank_bases,
    torch::Tensor first_rank, int64_t B, int64_t n_samples, double tau, double eps, bool fci,
    bool write_g, torch::Tensor r_out = torch::Tensor(), torch::Tensor adj_x = torch::Tensor(),
    torch::Tensor union_marks = torch::Tensor(), torch::Tensor rows = torch::Tensor(),
    torch::Tensor hit_flags = torch::Tensor(), torch::Tensor min_abs = torch::Tensor()) {
    TORCH_CHECK(C.is_cuda() && G.is_cuda() && nbrs.is_cuda() && px.is_cuda(), "ci_from_comb CUDA");
    TORCH_CHECK(G.scalar_type() == at::kChar, "G must be int8");
    TORCH_CHECK(C.scalar_type() == at::kDouble ||
                    (WHYDRA_ENABLE_FLOAT32 && C.scalar_type() == at::kFloat),
                "C float64");
    TORCH_CHECK(k >= 0 && k <= kMaxFused, "fused CI supports k=0..32");
    const int64_t E = px.numel();
    TORCH_CHECK(offsets.numel() == E + 1, "offsets length E+1");
    TORCH_CHECK(first_rank.numel() == E && first_rank.scalar_type() == at::kLong, "first_rank");
    if (E <= 0 || B <= 0) {
        return;
    }
    const int max_d = static_cast<int>(nbrs.size(1));
    const int64_t* rbp =
        rank_bases.defined() && rank_bases.numel() == E ? rank_bases.data_ptr<int64_t>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int8_t* adjptr = nullptr;
    int64_t adj_s0 = 0;
    int64_t adj_s1 = 1;
    if (adj_x.defined() && adj_x.numel() > 0) {
        TORCH_CHECK(adj_x.is_cuda() && adj_x.scalar_type() == at::kChar, "adj_x int8 CUDA");
        TORCH_CHECK(adj_x.is_contiguous(), "adj_x contiguous");
        const int64_t p = C.size(0);
        if (adj_x.dim() == 1) {
            TORCH_CHECK(adj_x.numel() == p, "adj_x length p");
            adj_s0 = 0;
            adj_s1 = 1;
        } else {
            TORCH_CHECK(adj_x.dim() == 2 && adj_x.size(0) == p && adj_x.size(1) == p,
                        "adj_x [p] or [p,p]");
            adj_s0 = adj_x.stride(0);
            adj_s1 = adj_x.stride(1);
        }
        adjptr = adj_x.data_ptr<int8_t>();
    }
    WHYDRA_DISPATCH_FLOATING(C.scalar_type(), "ci_from_comb", [&] {
        scalar_t* rptr = nullptr;
        if (r_out.defined() && r_out.numel() == B) {
            TORCH_CHECK(r_out.scalar_type() == C.scalar_type(), "r_out dtype");
            rptr = r_out.data_ptr<scalar_t>();
        }
        uint8_t* uptr = nullptr;
        const int* rowptr = nullptr;
        int64_t u_stride = 0;
        uint8_t* hptr = nullptr;
        const int64_t n_pairs =
            union_marks.defined() && union_marks.dim() == 2
                ? union_marks.size(0)
                : (hit_flags.defined() && hit_flags.numel() > 0 ? hit_flags.numel() : 0);
        if (union_marks.defined() && union_marks.numel() > 0) {
            TORCH_CHECK(union_marks.is_cuda() && union_marks.scalar_type() == at::kByte,
                        "union_marks uint8 CUDA");
            TORCH_CHECK(union_marks.is_contiguous(), "union_marks contiguous");
            uptr = union_marks.data_ptr<uint8_t>();
            u_stride = union_marks.size(1);
        }
        if (rows.defined() && rows.numel() > 0) {
            TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kInt, "rows int32 CUDA");
            TORCH_CHECK(rows.numel() == E, "rows length E");
            rowptr = rows.data_ptr<int>();
        }
        if (hit_flags.defined() && hit_flags.numel() > 0) {
            TORCH_CHECK(hit_flags.is_cuda() && hit_flags.scalar_type() == at::kByte,
                        "hit_flags uint8 CUDA");
            TORCH_CHECK(hit_flags.numel() == n_pairs, "hit_flags length matches union rows");
            hptr = hit_flags.data_ptr<uint8_t>();
        }
        unsigned long long* mptr = nullptr;
        if (min_abs.defined() && min_abs.numel() >= E) {
            TORCH_CHECK(min_abs.is_cuda() && min_abs.scalar_type() == at::kLong,
                        "min_abs int64 CUDA");
            mptr = reinterpret_cast<unsigned long long*>(min_abs.data_ptr<int64_t>());
        }
        // ``rows == nullptr`` means ``row = e`` (single-wave: one directed
        // pair per row), which the kernel handles natively; a materialized
        // ``rows`` is only needed when a stripe compacts the live pairs.
        if (uptr != nullptr || hptr != nullptr) {
            TORCH_CHECK(rowptr != nullptr || hptr == nullptr || n_pairs == E,
                        "union marking rows/hit_flags length");
        }
        auto launch = [&](auto kernel) {
            kernel<<<grid(B), kBlock, 0, stream>>>(
                C.data_ptr<scalar_t>(), C.stride(0), C.stride(1), G.data_ptr<int8_t>(), G.stride(0),
                G.stride(1), nbrs.data_ptr<int>(), deg.data_ptr<int>(), px.data_ptr<int>(),
                py.data_ptr<int>(), offsets.data_ptr<int64_t>(), rbp, rank_base,
                first_rank.data_ptr<int64_t>(), E, B, max_d, static_cast<int>(k),
                static_cast<int>(n_samples), tau, fci ? 1 : 0, write_g ? 1 : 0,
                static_cast<scalar_t>(eps), rptr, adjptr, adj_s0, adj_s1, uptr, rowptr, u_stride,
                hptr, mptr);
        };
        if (k <= 3) {
            launch(ci_from_comb_kernel<scalar_t, 4>);
        } else {
            launch(ci_from_comb_kernel<scalar_t, kMaxFused>);
        }
    });
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

// Runtime switch: the pair-major kernel for k <= 3 (and the k=0 x<y symmetry
// reduction) is the default. ``WHYDRA_PC_SMALLK=0`` restores the flat
// offset-search kernel for A/B; the equivalence of the two paths is covered by
// tests/test_cuda_pc_sepset_union.py.
bool pc_smallk_enabled() {
    const char* s = std::getenv("WHYDRA_PC_SMALLK");
    if (s == nullptr) {
        return true;
    }
    return !(s[0] == '0' && s[1] == '\0');
}

// ``WHYDRA_PC_WORKSPACE=0`` disables the cached per-device depth workspace and
// restores the per-depth allocation path (A/B switch).
bool pc_workspace_enabled() {
    const char* s = std::getenv("WHYDRA_PC_WORKSPACE");
    if (s == nullptr) {
        return true;
    }
    return !(s[0] == '0' && s[1] == '\0');
}

// Host-side saturating binomial for the hub fallback heuristic.
int64_t binom_host(int n, int k) {
    if (k < 0 || n < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n - k) {
        k = n - k;
    }
    int64_t r = 1;
    for (int i = 1; i <= k; ++i) {
        int64_t m = static_cast<int64_t>(n - k + i);
        if (r > std::numeric_limits<int64_t>::max() / m) {
            return std::numeric_limits<int64_t>::max();
        }
        r *= m;
        r /= i;
    }
    return r;
}

// Persistent per-device buffers for the PC union depth. Sizes are grown only
// when a problem needs more than the cached capacity; the buffers are then
// reused by every depth. ``tab`` stays all -1 between depths because the fused
// commit resets the entries it consumes.
//
// Invariants:
//   * ``px``/``py``/``ncomb``/``first`` have capacity ``cap >= p * max_d``,
//     an upper bound on the number of live directed pairs.
//   * ``union_marks`` is [cap, p] uint8, ``hit_flags`` [cap]; both are only
//     allocated for k >= 1 (k = 0 needs no member bits).
//   * ``tab`` is 2 * p*p int32: two halves (tab_a, tab_b) for the two
//     directions of each canonical key. It is all -1 after every fused commit.
//   * ``pinned`` is a small host staging page for the few counters copied
//     back per depth, so the driver needs only the CI-grid sync, not one sync
//     per scalar.
struct PcDepthWorkspace {
    int p = -1;
    int64_t cap = 0;
    torch::Tensor px;
    torch::Tensor py;
    torch::Tensor ncomb;
    torch::Tensor first;
    torch::Tensor E_out;
    torch::Tensor n_tests;
    torch::Tensor union_marks;
    torch::Tensor hit_flags;
    torch::Tensor tab;
    torch::Tensor overflow;
    torch::Tensor empty_i64;
    torch::Tensor pinned;
    cudaEvent_t e_event = nullptr;
};

PcDepthWorkspace& pc_depth_workspace(int device) {
    static auto* cache = new std::unordered_map<int, PcDepthWorkspace>();
    static auto* lock = new std::mutex();
    std::lock_guard<std::mutex> guard(*lock);
    return (*cache)[device];
}

// Reallocate ``w`` to hold ``cap`` live pairs for a ``p``-vertex graph: the
// workspace is re-created when ``p`` changes or ``cap`` grows (capacity may
// therefore shrink on a p change). The union buffers are re-created lazily on
// the next k >= 1 depth because their shapes depend on both p and cap.
void pc_depth_workspace_ensure(PcDepthWorkspace& w, int p, int64_t cap, bool need_union,
                               const torch::TensorOptions& i32, const torch::TensorOptions& i64,
                               const torch::TensorOptions& u8) {
    const bool realloc = w.p != p || w.cap < cap;
    if (w.e_event == nullptr) {
        cudaEventCreateWithFlags(&w.e_event, cudaEventDisableTiming);
    }
    if (realloc) {
        w.p = p;
        w.cap = cap;
        w.px = torch::empty({cap}, i32);
        w.py = torch::empty({cap}, i32);
        w.ncomb = torch::empty({cap}, i64);
        w.first = torch::empty({cap}, i64);
        w.E_out = torch::empty({1}, i32);
        w.n_tests = torch::empty({1}, i64);
        w.overflow = torch::empty({1}, i32);
        w.empty_i64 = torch::empty({0}, i64);
        // Invalidate the union buffers: their p-dependent shapes changed.
        w.union_marks = torch::Tensor();
        w.hit_flags = torch::Tensor();
        w.tab = torch::Tensor();
        if (!w.pinned.defined()) {
            w.pinned =
                torch::empty({4}, torch::TensorOptions().dtype(at::kLong).pinned_memory(true));
        }
    }
    if (need_union && !w.union_marks.defined()) {
        w.union_marks = torch::empty({cap, static_cast<int64_t>(p)}, u8);
        w.hit_flags = torch::empty({cap}, u8);
        w.tab = torch::full({2 * static_cast<int64_t>(p) * p}, -1, i32);
    }
}

// Pair-major replacement for ``graph_batch_ci_from_comb`` on the PC-stable
// union skeleton path, k = 0..3. Same math / bookkeeping; ``blockIdx`` selects
// the live directed pair, threads stride its local ranks. Requires the global
// rank base to be zero (the PC depth driver always passes rank_base = 0).
void graph_batch_ci_from_comb_pair_major(torch::Tensor C, torch::Tensor G, torch::Tensor nbrs,
                                         torch::Tensor deg, torch::Tensor px, torch::Tensor py,
                                         torch::Tensor offsets, torch::Tensor ncomb, int64_t k,
                                         torch::Tensor first_rank, int64_t E, int64_t n_samples,
                                         double tau, double eps, bool write_g, torch::Tensor r_out,
                                         torch::Tensor union_marks, torch::Tensor rows,
                                         torch::Tensor hit_flags, bool fci, torch::Tensor min_abs) {
    TORCH_CHECK(k >= 0 && k <= 3, "pair-major kernel is k=0..3");
    TORCH_CHECK(C.is_cuda() && G.is_cuda() && nbrs.is_cuda() && px.is_cuda(), "pair-major CUDA");
    TORCH_CHECK(G.scalar_type() == at::kChar, "G int8");
    TORCH_CHECK(ncomb.is_cuda() && ncomb.scalar_type() == at::kLong, "ncomb int64");
    if (E <= 0) {
        return;
    }
    const int max_d = static_cast<int>(nbrs.size(1));
    TORCH_CHECK(max_d > 0, "empty adjacency");
    const bool have_r = r_out.defined() && r_out.numel() > 0;
    TORCH_CHECK(!have_r || (offsets.defined() && offsets.numel() > E), "r_out needs offsets");
    uint8_t* uptr = nullptr;
    const int* rowptr = nullptr;
    int64_t u_stride = 0;
    uint8_t* hptr = nullptr;
    if (union_marks.defined() && union_marks.numel() > 0) {
        TORCH_CHECK(union_marks.is_cuda() && union_marks.scalar_type() == at::kByte &&
                        union_marks.is_contiguous(),
                    "union_marks uint8 CUDA contiguous");
        uptr = union_marks.data_ptr<uint8_t>();
        u_stride = union_marks.size(1);
    }
    const int64_t n_pairs = uptr != nullptr ? union_marks.size(0) : E;
    if (rows.defined() && rows.numel() > 0) {
        TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kInt && rows.numel() == E,
                    "rows int32 length E");
        rowptr = rows.data_ptr<int>();
    }
    if (hit_flags.defined() && hit_flags.numel() > 0) {
        TORCH_CHECK(hit_flags.is_cuda() && hit_flags.scalar_type() == at::kByte &&
                        hit_flags.numel() == n_pairs,
                    "hit_flags uint8");
        hptr = hit_flags.data_ptr<uint8_t>();
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t* offptr =
        offsets.defined() && offsets.numel() > E ? offsets.data_ptr<int64_t>() : nullptr;
    unsigned long long* mptr = nullptr;
    if (min_abs.defined() && min_abs.numel() >= E) {
        TORCH_CHECK(min_abs.is_cuda() && min_abs.scalar_type() == at::kLong, "min_abs int64 CUDA");
        mptr = reinterpret_cast<unsigned long long*>(min_abs.data_ptr<int64_t>());
    }
    const int ga = fci ? 1 : 0;
    WHYDRA_DISPATCH_FLOATING(C.scalar_type(), "ci_pair_major", [&] {
        scalar_t* rptr = have_r ? r_out.data_ptr<scalar_t>() : nullptr;
        if (k == 0) {
            const int64_t nblk = (E + kBlock - 1) / kBlock;
            ci_pair_major_k0_kernel<scalar_t><<<static_cast<unsigned>(nblk), kBlock, 0, stream>>>(
                C.data_ptr<scalar_t>(), C.stride(0), C.stride(1), G.data_ptr<int8_t>(), G.stride(0),
                G.stride(1), px.data_ptr<int>(), py.data_ptr<int>(), first_rank.data_ptr<int64_t>(),
                E, static_cast<int>(n_samples), tau, ga, write_g ? 1 : 0,
                static_cast<scalar_t>(eps), rptr, offptr, uptr, rowptr, u_stride, hptr, mptr);
            return;
        }
        const int threads = (k == 1) ? 32 : (k == 2 ? 128 : 256);
        const int64_t gx_cap = 1 << 20;
        const int64_t gx = E < gx_cap ? E : gx_cap;
        const int64_t gy = (E + gx - 1) / gx;
        dim3 grid2(static_cast<unsigned>(gx), static_cast<unsigned>(gy));
        if (k == 1) {
            ci_pair_major_kernel<scalar_t, 4><<<grid2, threads, 0, stream>>>(
                C.data_ptr<scalar_t>(), C.stride(0), C.stride(1), G.data_ptr<int8_t>(), G.stride(0),
                G.stride(1), nbrs.data_ptr<int>(), deg.data_ptr<int>(), px.data_ptr<int>(),
                py.data_ptr<int>(), offptr, ncomb.data_ptr<int64_t>(),
                first_rank.data_ptr<int64_t>(), E, max_d, 1, static_cast<int>(n_samples), tau, ga,
                write_g ? 1 : 0, static_cast<scalar_t>(eps), rptr, nullptr, 0, 1, uptr, rowptr,
                u_stride, hptr, mptr);
        } else if (k == 2) {
            ci_pair_major_kernel<scalar_t, 4><<<grid2, threads, 0, stream>>>(
                C.data_ptr<scalar_t>(), C.stride(0), C.stride(1), G.data_ptr<int8_t>(), G.stride(0),
                G.stride(1), nbrs.data_ptr<int>(), deg.data_ptr<int>(), px.data_ptr<int>(),
                py.data_ptr<int>(), offptr, ncomb.data_ptr<int64_t>(),
                first_rank.data_ptr<int64_t>(), E, max_d, 2, static_cast<int>(n_samples), tau, ga,
                write_g ? 1 : 0, static_cast<scalar_t>(eps), rptr, nullptr, 0, 1, uptr, rowptr,
                u_stride, hptr, mptr);
        } else {
            ci_pair_major_kernel<scalar_t, 4><<<grid2, threads, 0, stream>>>(
                C.data_ptr<scalar_t>(), C.stride(0), C.stride(1), G.data_ptr<int8_t>(), G.stride(0),
                G.stride(1), nbrs.data_ptr<int>(), deg.data_ptr<int>(), px.data_ptr<int>(),
                py.data_ptr<int>(), offptr, ncomb.data_ptr<int64_t>(),
                first_rank.data_ptr<int64_t>(), E, max_d, 3, static_cast<int>(n_samples), tau, ga,
                write_g ? 1 : 0, static_cast<scalar_t>(eps), rptr, nullptr, 0, 1, uptr, rowptr,
                u_stride, hptr, mptr);
        }
    });
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__global__ void prefix_pmax_kernel(const T* __restrict__ r, const int64_t* __restrict__ offsets,
                                   const int64_t* __restrict__ first_rank,
                                   const int* __restrict__ px, const int* __restrict__ py,
                                   double* __restrict__ pmax, int64_t ps0, int64_t ps1, int64_t B,
                                   int64_t E, int n_samples, int k, int64_t rank_max) {
    int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }
    double rv = static_cast<double>(r[b]);
    if (!isfinite(rv)) {
        return;
    }
    int64_t e = pair_index(offsets, E, b);
    int64_t local = b - offsets[e];
    int64_t fr = first_rank[e];
    if (fr != rank_max && local > fr) {
        return;
    }
    double pv = p_from_r(rv, n_samples, k);
    if (!isfinite(pv)) {
        return;
    }
    int x = px[e];
    int y = py[e];
    atomic_max_f64(&pmax[static_cast<int64_t>(x) * ps0 + static_cast<int64_t>(y) * ps1], pv);
    if (k == 0) {
        atomic_max_f64(&pmax[static_cast<int64_t>(y) * ps0 + static_cast<int64_t>(x) * ps1], pv);
    }
}

void graph_batch_prefix_pmax(torch::Tensor r, torch::Tensor offsets, torch::Tensor first_rank,
                             torch::Tensor px, torch::Tensor py, torch::Tensor pmax,
                             int64_t n_samples, int64_t k, int64_t rank_max) {
    TORCH_CHECK(r.is_cuda() && offsets.is_cuda() && pmax.is_cuda(), "prefix_pmax CUDA");
    TORCH_CHECK(pmax.scalar_type() == at::kDouble && pmax.dim() == 2, "pmax [p,p] float64");
    TORCH_CHECK(px.scalar_type() == at::kInt && py.scalar_type() == at::kInt, "px/py int32");
    TORCH_CHECK(first_rank.scalar_type() == at::kLong, "first_rank");
    const int64_t B = r.numel();
    const int64_t E = px.numel();
    if (B <= 0 || E <= 0) {
        return;
    }
    TORCH_CHECK(offsets.numel() == E + 1, "offsets length E+1");
    TORCH_CHECK(first_rank.numel() == E, "first_rank length E");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    WHYDRA_DISPATCH_FLOATING(r.scalar_type(), "prefix_pmax", [&] {
        prefix_pmax_kernel<scalar_t><<<grid(B), kBlock, 0, stream>>>(
            r.data_ptr<scalar_t>(), offsets.data_ptr<int64_t>(), first_rank.data_ptr<int64_t>(),
            px.data_ptr<int>(), py.data_ptr<int>(), pmax.data_ptr<double>(), pmax.stride(0),
            pmax.stride(1), B, E, static_cast<int>(n_samples), static_cast<int>(k), rank_max);
    });
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_first_hit(torch::Tensor r, torch::Tensor offsets, torch::Tensor first_rank,
                           int64_t n_samples, int64_t k, double tau, bool ge_alpha) {
    TORCH_CHECK(r.is_cuda() && offsets.is_cuda() && first_rank.is_cuda(), "first_hit CUDA");
    const int64_t B = r.numel();
    const int64_t E = first_rank.numel();
    TORCH_CHECK(offsets.numel() == E + 1, "offsets length E+1");
    if (B == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    WHYDRA_DISPATCH_FLOATING(r.scalar_type(), "graph_batch_first_hit", [&] {
        first_hit_kernel<scalar_t><<<grid(B), kBlock, 0, stream>>>(
            r.data_ptr<scalar_t>(), offsets.data_ptr<int64_t>(), first_rank.data_ptr<int64_t>(), B,
            E, static_cast<int>(n_samples), static_cast<int>(k), tau, ge_alpha ? 1 : 0);
    });
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_fill_at_ranks(torch::Tensor nbrs, torch::Tensor deg, torch::Tensor px,
                               torch::Tensor py, torch::Tensor ranks, int64_t k, torch::Tensor x,
                               torch::Tensor y, torch::Tensor S) {
    TORCH_CHECK(nbrs.is_cuda() && x.is_cuda() && ranks.is_cuda(), "fill_at_ranks CUDA");
    TORCH_CHECK(k >= 0 && k <= kMaxCond, "k out of range");
    const int64_t E = px.numel();
    TORCH_CHECK(x.numel() == E && y.numel() == E && ranks.numel() == E, "E mismatch");
    if (k > 0) {
        TORCH_CHECK(S.dim() == 2 && S.size(0) == E && S.size(1) == k, "S (E, k)");
    }
    if (E == 0) {
        return;
    }
    const int max_d = static_cast<int>(nbrs.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int* Sptr = k > 0 ? S.data_ptr<int>() : nullptr;
    fill_at_ranks_kernel<<<grid(E), kBlock, 0, stream>>>(
        nbrs.data_ptr<int>(), deg.data_ptr<int>(), px.data_ptr<int>(), py.data_ptr<int>(),
        ranks.data_ptr<int64_t>(), E, max_d, static_cast<int>(k), x.data_ptr<int>(),
        y.data_ptr<int>(), Sptr);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_claim_sepsets(torch::Tensor px, torch::Tensor py, torch::Tensor first_rank,
                               torch::Tensor rank_bases, int64_t rank_base, int64_t rank_max,
                               torch::Tensor nbrs, torch::Tensor deg, torch::Tensor ranges,
                               torch::Tensor nnz, torch::Tensor S, torch::Tensor overflow,
                               int64_t k) {
    TORCH_CHECK(px.is_cuda() && first_rank.is_cuda() && ranges.is_cuda(), "claim_sepsets CUDA");
    TORCH_CHECK(nnz.is_cuda() && S.is_cuda(), "claim buffers CUDA");
    TORCH_CHECK(nnz.numel() == 1, "nnz scalar");
    TORCH_CHECK(overflow.is_cuda() && overflow.numel() == 1 && overflow.scalar_type() == at::kInt,
                "overflow scalar int32 CUDA");
    TORCH_CHECK(ranges.dim() == 3 && ranges.size(2) == 2, "ranges (p, p, 2)");
    TORCH_CHECK(ranges.size(0) == ranges.size(1), "ranges square");
    TORCH_CHECK(k >= 0 && k <= kMaxCond, "k out of range");
    const int64_t E = px.numel();
    TORCH_CHECK(py.numel() == E && first_rank.numel() == E, "E mismatch");
    if (E == 0) {
        return;
    }
    const int p = static_cast<int>(ranges.size(0));
    const int max_d = static_cast<int>(nbrs.size(1));
    const int64_t s_cap = S.numel();
    const int64_t* rb = rank_bases.numel() > 0 ? rank_bases.data_ptr<int64_t>() : nullptr;
    if (rank_bases.numel() > 0) {
        TORCH_CHECK(rank_bases.numel() == E, "rank_bases E");
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    claim_sepsets_kernel<<<grid(E), kBlock, 0, stream>>>(
        px.data_ptr<int>(), py.data_ptr<int>(), first_rank.data_ptr<int64_t>(), rb, rank_base,
        rank_max, nbrs.data_ptr<int>(), deg.data_ptr<int>(), E, p, max_d, static_cast<int>(k),
        s_cap, ranges.data_ptr<int64_t>(), nnz.data_ptr<int64_t>(), S.data_ptr<int64_t>(), nullptr,
        0, 0, 0, overflow.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_pdsep_commit_walk(torch::Tensor py, torch::Tensor hit_k, torch::Tensor hit_rank,
                                   torch::Tensor row_off, torch::Tensor nbrs, torch::Tensor deg,
                                   torch::Tensor ranges, torch::Tensor nnz, torch::Tensor S,
                                   torch::Tensor overflow) {
    TORCH_CHECK(py.is_cuda() && hit_k.is_cuda() && hit_rank.is_cuda(), "pdsep_commit CUDA");
    TORCH_CHECK(row_off.is_cuda() && nbrs.is_cuda() && deg.is_cuda(), "pdsep_commit adj CUDA");
    TORCH_CHECK(ranges.is_cuda() && nnz.is_cuda() && S.is_cuda(), "pdsep_commit csr CUDA");
    TORCH_CHECK(nnz.numel() == 1, "nnz scalar");
    TORCH_CHECK(overflow.is_cuda() && overflow.numel() == 1 && overflow.scalar_type() == at::kInt,
                "overflow scalar int32 CUDA");
    TORCH_CHECK(ranges.dim() == 3 && ranges.size(2) == 2, "ranges (p, p, 2)");
    TORCH_CHECK(ranges.size(0) == ranges.size(1), "ranges square");
    const int p = static_cast<int>(ranges.size(0));
    TORCH_CHECK(row_off.numel() == p + 1, "row_off p+1");
    TORCH_CHECK(hit_k.scalar_type() == at::kInt, "hit_k int32");
    TORCH_CHECK(hit_rank.scalar_type() == at::kLong, "hit_rank int64");
    TORCH_CHECK(py.scalar_type() == at::kInt, "py int32");
    const int64_t E = py.numel();
    TORCH_CHECK(hit_k.numel() == E && hit_rank.numel() == E, "E mismatch");
    if (p == 0) {
        return;
    }
    const int max_d = static_cast<int>(nbrs.size(1));
    const int64_t s_cap = S.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pdsep_commit_walk_kernel<<<1, kBlock, 0, stream>>>(
        py.data_ptr<int>(), hit_k.data_ptr<int>(), hit_rank.data_ptr<int64_t>(),
        row_off.data_ptr<int64_t>(), nbrs.data_ptr<int>(), deg.data_ptr<int>(), p, max_d, s_cap,
        ranges.data_ptr<int64_t>(), nnz.data_ptr<int64_t>(), S.data_ptr<int64_t>(),
        overflow.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_hits_from_r(torch::Tensor r, torch::Tensor hit, int64_t n_samples, int64_t k,
                             double tau, bool ge_alpha) {
    TORCH_CHECK(r.is_cuda() && hit.is_cuda(), "hits_from_r CUDA");
    const int64_t B = r.numel();
    TORCH_CHECK(hit.numel() == B && hit.scalar_type() == at::kByte, "hit uint8 [B]");
    if (B == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    WHYDRA_DISPATCH_FLOATING(r.scalar_type(), "graph_batch_hits_from_r", [&] {
        hits_from_r_kernel<scalar_t><<<grid(B), kBlock, 0, stream>>>(
            r.data_ptr<scalar_t>(), hit.data_ptr<uint8_t>(), B, static_cast<int>(n_samples),
            static_cast<int>(k), tau, ge_alpha ? 1 : 0);
    });
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_apply_hits(torch::Tensor G, torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(G.is_cuda() && x.is_cuda() && y.is_cuda(), "apply_hits CUDA");
    TORCH_CHECK(G.dim() == 2 && G.size(0) == G.size(1), "G must be square");
    TORCH_CHECK(G.scalar_type() == at::kChar, "G must be int8");
    TORCH_CHECK(x.scalar_type() == at::kInt && y.scalar_type() == at::kInt, "x/y int32");
    TORCH_CHECK(x.numel() == y.numel(), "x/y length");
    const int64_t n = x.numel();
    if (n == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    apply_hits_kernel<<<grid(n), kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), G.stride(0), G.stride(1), x.data_ptr<int>(), y.data_ptr<int>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_edge_present(torch::Tensor G, torch::Tensor x, torch::Tensor y,
                              torch::Tensor present) {
    TORCH_CHECK(G.is_cuda() && x.is_cuda() && y.is_cuda() && present.is_cuda(),
                "edge_present CUDA");
    TORCH_CHECK(G.dim() == 2 && G.size(0) == G.size(1), "G must be square");
    TORCH_CHECK(G.scalar_type() == at::kChar, "G must be int8");
    TORCH_CHECK(x.scalar_type() == at::kInt && y.scalar_type() == at::kInt, "x/y int32");
    TORCH_CHECK(present.scalar_type() == at::kByte, "present uint8");
    TORCH_CHECK(x.numel() == y.numel() && present.numel() == x.numel(), "x/y/present length");
    const int64_t n = x.numel();
    if (n == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    edge_present_kernel<<<grid(n), kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), G.stride(0), G.stride(1), x.data_ptr<int>(), y.data_ptr<int>(),
        present.data_ptr<uint8_t>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_delete_if_indep(torch::Tensor G, torch::Tensor x, torch::Tensor y,
                                 torch::Tensor indep) {
    TORCH_CHECK(G.is_cuda() && x.is_cuda() && y.is_cuda() && indep.is_cuda(),
                "delete_if_indep CUDA");
    TORCH_CHECK(G.dim() == 2 && G.size(0) == G.size(1), "G must be square");
    TORCH_CHECK(G.scalar_type() == at::kChar, "G must be int8");
    TORCH_CHECK(x.scalar_type() == at::kInt && y.scalar_type() == at::kInt, "x/y int32");
    TORCH_CHECK(indep.scalar_type() == at::kByte, "indep uint8");
    TORCH_CHECK(x.numel() == y.numel() && indep.numel() == x.numel(), "x/y/indep length");
    const int64_t n = x.numel();
    if (n == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    delete_if_indep_kernel<<<grid(n), kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), G.stride(0), G.stride(1), x.data_ptr<int>(), y.data_ptr<int>(),
        indep.data_ptr<uint8_t>(), n);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_pack_unique(torch::Tensor px, torch::Tensor py, torch::Tensor px_out,
                             torch::Tensor py_out, torch::Tensor ranks_out, torch::Tensor ncomb_out,
                             torch::Tensor E_out) {
    TORCH_CHECK(px.is_cuda() && E_out.is_cuda(), "pack_unique CUDA");
    TORCH_CHECK(E_out.numel() == 1 && E_out.scalar_type() == at::kInt, "E_out int32[1]");
    const int64_t n = px.numel();
    if (n == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pack_px_lt_py_kernel<<<grid(n), kBlock, 0, stream>>>(
        px.data_ptr<int>(), py.data_ptr<int>(), n, px_out.data_ptr<int>(), py_out.data_ptr<int>(),
        ranks_out.data_ptr<int64_t>(), ncomb_out.data_ptr<int64_t>(), E_out.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_pack_ncomb(torch::Tensor px, torch::Tensor py, torch::Tensor ncomb,
                            torch::Tensor px_out, torch::Tensor py_out, torch::Tensor ranks_out,
                            torch::Tensor ncomb_out, torch::Tensor E_out) {
    TORCH_CHECK(px.is_cuda() && E_out.is_cuda(), "pack_ncomb CUDA");
    const int64_t n = px.numel();
    if (n == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pack_ncomb_pos_kernel<<<grid(n), kBlock, 0, stream>>>(
        px.data_ptr<int>(), py.data_ptr<int>(), ncomb.data_ptr<int64_t>(), n,
        px_out.data_ptr<int>(), py_out.data_ptr<int>(), ranks_out.data_ptr<int64_t>(),
        ncomb_out.data_ptr<int64_t>(), E_out.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_compact_live(torch::Tensor px, torch::Tensor py, torch::Tensor ranks,
                              torch::Tensor ncomb, torch::Tensor hit, torch::Tensor S, int64_t k,
                              int64_t E, torch::Tensor px_out, torch::Tensor py_out,
                              torch::Tensor ranks_out, torch::Tensor ncomb_out, torch::Tensor hit_x,
                              torch::Tensor hit_y, torch::Tensor hit_S, torch::Tensor E_out,
                              torch::Tensor H_out) {
    TORCH_CHECK(px.is_cuda() && hit.is_cuda() && E_out.is_cuda(), "compact_live CUDA");
    TORCH_CHECK(E_out.numel() == 1 && H_out.numel() == 1, "E_out/H_out scalars");
    if (E <= 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int* Sptr = (k > 0 && S.numel() > 0) ? S.data_ptr<int>() : nullptr;
    int* hitSptr = (k > 0 && hit_S.numel() > 0) ? hit_S.data_ptr<int>() : nullptr;
    compact_live_kernel<<<grid(E), kBlock, 0, stream>>>(
        px.data_ptr<int>(), py.data_ptr<int>(), ranks.data_ptr<int64_t>(),
        ncomb.data_ptr<int64_t>(), hit.data_ptr<uint8_t>(), Sptr, static_cast<int>(k), E,
        px_out.data_ptr<int>(), py_out.data_ptr<int>(), ranks_out.data_ptr<int64_t>(),
        ncomb_out.data_ptr<int64_t>(), hit_x.data_ptr<int>(), hit_y.data_ptr<int>(), hitSptr,
        E_out.data_ptr<int>(), H_out.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_mark_s_k1(torch::Tensor nbrs, torch::Tensor deg, torch::Tensor mark) {
    TORCH_CHECK(nbrs.is_cuda() && deg.is_cuda() && mark.is_cuda(), "mark_s_k1 CUDA");
    const int p = static_cast<int>(deg.numel());
    const int max_d = static_cast<int>(nbrs.size(1));
    TORCH_CHECK(mark.numel() == p, "mark_s_k1 size");
    if (p == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mark_s_k1_kernel<<<grid(p), kBlock, 0, stream>>>(nbrs.data_ptr<int>(), deg.data_ptr<int>(), p,
                                                     max_d, mark.data_ptr<uint8_t>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_mark_s_k2(torch::Tensor nbrs, torch::Tensor deg, torch::Tensor mark) {
    TORCH_CHECK(nbrs.is_cuda() && deg.is_cuda() && mark.is_cuda(), "mark_s_k2 CUDA");
    const int p = static_cast<int>(deg.numel());
    const int max_d = static_cast<int>(nbrs.size(1));
    TORCH_CHECK(mark.numel() == static_cast<int64_t>(p) * p, "mark_s_k2 size");
    if (p == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mark_s_k2_kernel<<<grid(p), kBlock, 0, stream>>>(nbrs.data_ptr<int>(), deg.data_ptr<int>(), p,
                                                     max_d, mark.data_ptr<uint8_t>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_mark_s_ranks(torch::Tensor nbrs, torch::Tensor deg, int64_t k,
                              torch::Tensor bits) {
    TORCH_CHECK(nbrs.is_cuda() && deg.is_cuda() && bits.is_cuda(), "mark_s_ranks CUDA");
    TORCH_CHECK(k >= 1 && k <= kMaxCond, "k out of range");
    const int p = static_cast<int>(deg.numel());
    const int max_d = static_cast<int>(nbrs.size(1));
    const int64_t ncomb = bits.numel();
    if (p == 0 || ncomb == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mark_s_ranks_kernel<<<grid(p), kBlock, 0, stream>>>(nbrs.data_ptr<int>(), deg.data_ptr<int>(),
                                                        p, max_d, static_cast<int>(k),
                                                        bits.data_ptr<uint8_t>(), ncomb);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_unrank_rows(torch::Tensor idx, int64_t p, int64_t k, torch::Tensor S) {
    TORCH_CHECK(idx.is_cuda() && S.is_cuda(), "unrank_rows CUDA");
    TORCH_CHECK(idx.scalar_type() == at::kLong, "idx int64");
    const int64_t U = idx.numel();
    TORCH_CHECK(S.size(0) == U && S.size(1) == k, "S shape");
    if (U == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    unrank_rows_kernel<<<grid(U), kBlock, 0, stream>>>(
        idx.data_ptr<int64_t>(), U, static_cast<int>(p), static_cast<int>(k), S.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_count_s_tests(torch::Tensor G, torch::Tensor nbrs, torch::Tensor deg,
                               torch::Tensor unique_S, int64_t undirected, torch::Tensor counts) {
    TORCH_CHECK(G.is_cuda() && nbrs.is_cuda() && unique_S.is_cuda() && counts.is_cuda(),
                "count_s_tests CUDA");
    TORCH_CHECK(G.scalar_type() == at::kChar, "G must be int8");
    TORCH_CHECK(unique_S.is_contiguous() && counts.is_contiguous(), "contiguous");
    const int64_t U = unique_S.size(0);
    const int k = unique_S.size(0) == 0 ? 0 : static_cast<int>(unique_S.size(1));
    TORCH_CHECK(k >= 1 && k <= kMaxCond, "k out of range");
    TORCH_CHECK(counts.numel() == U, "counts size");
    if (U == 0) {
        return;
    }
    const int max_d = static_cast<int>(nbrs.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    count_s_tests_kernel<<<grid(U), kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), G.stride(0), G.stride(1), nbrs.data_ptr<int>(), deg.data_ptr<int>(),
        unique_S.data_ptr<int>(), U, max_d, k, undirected ? 1 : 0, counts.data_ptr<int64_t>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_fill_s_tests(torch::Tensor G, torch::Tensor nbrs, torch::Tensor deg,
                              torch::Tensor unique_S, torch::Tensor offsets, int64_t undirected,
                              torch::Tensor x, torch::Tensor y, torch::Tensor S) {
    TORCH_CHECK(G.is_cuda() && unique_S.is_cuda() && x.is_cuda(), "fill_s_tests CUDA");
    const int64_t U = unique_S.size(0);
    const int k = U == 0 ? 0 : static_cast<int>(unique_S.size(1));
    if (U == 0) {
        return;
    }
    const int max_d = static_cast<int>(nbrs.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fill_s_tests_kernel<<<grid(U), kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), G.stride(0), G.stride(1), nbrs.data_ptr<int>(), deg.data_ptr<int>(),
        unique_S.data_ptr<int>(), offsets.data_ptr<int64_t>(), U, max_d, k, undirected ? 1 : 0,
        x.data_ptr<int>(), y.data_ptr<int>(), S.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_count_nbr_combos(torch::Tensor deg, int64_t k, torch::Tensor counts) {
    TORCH_CHECK(deg.is_cuda() && counts.is_cuda(), "count_nbr_combos CUDA");
    const int p = static_cast<int>(deg.numel());
    if (p == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    count_nbr_combos_kernel<<<grid(p), kBlock, 0, stream>>>(
        deg.data_ptr<int>(), p, static_cast<int>(k), counts.data_ptr<int64_t>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_fill_nbr_combos(torch::Tensor nbrs, torch::Tensor deg, torch::Tensor offsets,
                                 int64_t k, torch::Tensor S) {
    TORCH_CHECK(nbrs.is_cuda() && S.is_cuda(), "fill_nbr_combos CUDA");
    const int p = static_cast<int>(deg.numel());
    const int max_d = static_cast<int>(nbrs.size(1));
    if (p == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fill_nbr_combos_kernel<<<grid(p), kBlock, 0, stream>>>(
        nbrs.data_ptr<int>(), deg.data_ptr<int>(), offsets.data_ptr<int64_t>(), p, max_d,
        static_cast<int>(k), S.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

int graph_batch_k_max() { return kMaxCond; }

int graph_batch_fused_k_max() { return kMaxFused; }

void graph_batch_pc_live_pairs(torch::Tensor nbrs, torch::Tensor deg, int64_t k, torch::Tensor px,
                               torch::Tensor py, torch::Tensor ncomb, torch::Tensor E_out) {
    TORCH_CHECK(nbrs.is_cuda() && deg.is_cuda(), "pc_live_pairs CUDA");
    TORCH_CHECK(nbrs.scalar_type() == at::kInt && deg.scalar_type() == at::kInt, "nbrs/deg int32");
    TORCH_CHECK(px.scalar_type() == at::kInt && py.scalar_type() == at::kInt, "px/py int32");
    TORCH_CHECK(ncomb.scalar_type() == at::kLong, "ncomb int64");
    TORCH_CHECK(E_out.numel() == 1 && E_out.scalar_type() == at::kInt, "E_out int32 scalar");
    TORCH_CHECK(px.numel() == py.numel() && px.numel() == ncomb.numel(), "output sizes");
    TORCH_CHECK(deg.numel() == nbrs.size(0), "deg length");
    const int p = static_cast<int>(deg.numel());
    const int max_d = nbrs.dim() == 2 ? static_cast<int>(nbrs.size(1)) : 0;
    TORCH_CHECK(k >= 0, "k must be >= 0");
    if (p == 0 || max_d == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pc_live_pairs_kernel<<<grid(static_cast<int64_t>(p) * max_d), kBlock, 0, stream>>>(
        nbrs.data_ptr<int>(), deg.data_ptr<int>(), p, max_d, static_cast<int>(k),
        px.data_ptr<int>(), py.data_ptr<int>(), ncomb.data_ptr<int64_t>(), E_out.data_ptr<int>(),
        nullptr, 0, nullptr, 0, 0);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_zero_flagged(torch::Tensor G, torch::Tensor px, torch::Tensor py,
                              torch::Tensor flags) {
    TORCH_CHECK(G.is_cuda() && px.is_cuda() && py.is_cuda() && flags.is_cuda(),
                "zero_flagged CUDA");
    TORCH_CHECK(G.scalar_type() == at::kChar && G.dim() == 2, "G int8 square");
    TORCH_CHECK(px.scalar_type() == at::kInt && py.scalar_type() == at::kInt, "px/py int32");
    TORCH_CHECK(flags.scalar_type() == at::kByte, "flags uint8");
    const int64_t E = px.numel();
    TORCH_CHECK(py.numel() == E && flags.numel() == E, "E mismatch");
    if (E == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    zero_flagged_kernel<<<grid(E), kBlock, 0, stream>>>(
        G.data_ptr<int8_t>(), G.stride(0), G.stride(1), px.data_ptr<int>(), py.data_ptr<int>(),
        flags.data_ptr<uint8_t>(), E);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_commit_union_mark(torch::Tensor px, torch::Tensor py, torch::Tensor flags,
                                   torch::Tensor tab_a, torch::Tensor tab_b, int64_t p) {
    TORCH_CHECK(px.is_cuda() && py.is_cuda() && flags.is_cuda(), "commit_union_mark CUDA");
    TORCH_CHECK(px.scalar_type() == at::kInt && py.scalar_type() == at::kInt, "px/py int32");
    TORCH_CHECK(flags.scalar_type() == at::kByte, "flags uint8");
    TORCH_CHECK(tab_a.is_cuda() && tab_b.is_cuda(), "tables CUDA");
    TORCH_CHECK(tab_a.scalar_type() == at::kInt && tab_b.scalar_type() == at::kInt, "tables int32");
    const int64_t E = px.numel();
    TORCH_CHECK(py.numel() == E && flags.numel() == E, "E mismatch");
    if (E == 0) {
        return;
    }
    TORCH_CHECK(px.is_contiguous() && py.is_contiguous() && flags.is_contiguous(),
                "commit_union_mark inputs contiguous");
    TORCH_CHECK(tab_a.numel() == p * p && tab_b.numel() == p * p, "table sizes p*p");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    commit_union_mark_kernel<<<grid(E), kBlock, 0, stream>>>(
        px.data_ptr<int>(), py.data_ptr<int>(), flags.data_ptr<uint8_t>(), E, static_cast<int>(p),
        tab_a.data_ptr<int>(), tab_b.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void graph_batch_commit_union_write(torch::Tensor tab_a, torch::Tensor tab_b,
                                    torch::Tensor union_marks, torch::Tensor ranges,
                                    torch::Tensor nnz, torch::Tensor S, torch::Tensor overflow) {
    TORCH_CHECK(tab_a.is_cuda() && tab_b.is_cuda() && union_marks.is_cuda(),
                "commit_union_write CUDA");
    TORCH_CHECK(ranges.is_cuda() && nnz.is_cuda() && S.is_cuda(), "csr CUDA");
    TORCH_CHECK(tab_a.scalar_type() == at::kInt && tab_b.scalar_type() == at::kInt, "tables int32");
    TORCH_CHECK(union_marks.scalar_type() == at::kByte && union_marks.dim() == 2,
                "union_marks uint8 [E, p]");
    TORCH_CHECK(ranges.dim() == 3 && ranges.size(2) == 2, "ranges (p, p, 2)");
    TORCH_CHECK(ranges.size(0) == ranges.size(1), "ranges square");
    TORCH_CHECK(nnz.numel() == 1 && nnz.scalar_type() == at::kLong, "nnz scalar int64");
    TORCH_CHECK(overflow.numel() == 1 && overflow.scalar_type() == at::kInt, "overflow int32");
    TORCH_CHECK(union_marks.is_contiguous(), "union_marks contiguous");
    const int p = static_cast<int>(ranges.size(0));
    TORCH_CHECK(tab_a.numel() == static_cast<int64_t>(p) * p, "table size p*p");
    TORCH_CHECK(union_marks.size(0) == 0 || union_marks.size(1) == p, "union_marks [E,p]");
    TORCH_CHECK(S.scalar_type() == at::kLong, "S int64");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t nkey = static_cast<int64_t>(p) * p;
    commit_union_write_kernel<<<grid(nkey), kBlock, 0, stream>>>(
        tab_a.data_ptr<int>(), tab_b.data_ptr<int>(), union_marks.data_ptr<uint8_t>(),
        static_cast<int64_t>(union_marks.size(1)), p, ranges.data_ptr<int64_t>(),
        nnz.data_ptr<int64_t>(), S.data_ptr<int64_t>(), static_cast<int64_t>(S.numel()),
        overflow.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

// Whole PC-stable union depth driven from C++: live pairs, per-pair test
// counts, one fused pair-major CI launch, the two-launch union commit, and the
// G update. Returns a 3-element CPU int64 tensor
// ``{n_tests, nnz_after, overflow}`` so Python needs no device->host round
// trip of its own. The only synchronization on the workspace path is the
// live-pair count needed to size the CI/commit grids; every scalar is staged
// through pinned memory and copied back with one stream sync at the end.
//
// Invariants preserved relative to the flat path (tested by
// tests/test_cuda_pc_sepset_union.py):
//   * k = 0 enumerates the upper triangle and returns E_live tests (half the
//     flat directed count); the union is empty, sepsets are (0,0) in both cells.
//   * k >= 1 stays directed; sepsets are the union of every separating S of
//     the depth, one record per canonical pair, written to both cells.
//   * n_tests is the number of planned CI tests for the selected kernel:
//     sum C(deg(x)-1, k) over live directed pairs, or E_live at k = 0.
torch::Tensor graph_batch_pc_depth_union_fci(torch::Tensor C, torch::Tensor G, torch::Tensor nbrs,
                                             torch::Tensor deg, torch::Tensor ranges,
                                             torch::Tensor S, torch::Tensor nnz, double tau,
                                             double eps, int64_t n_samples, int64_t k, bool fci,
                                             torch::Tensor pmax, torch::Tensor min_abs) {
    TORCH_CHECK(G.is_cuda() && G.scalar_type() == at::kChar && G.dim() == 2, "G int8 square CUDA");
    TORCH_CHECK(C.is_cuda() && C.dim() == 2 && C.size(0) == C.size(1), "C square");
    TORCH_CHECK(nbrs.is_cuda() && deg.is_cuda(), "adj CUDA");
    TORCH_CHECK(nbrs.scalar_type() == at::kInt && deg.scalar_type() == at::kInt, "adj int32");
    TORCH_CHECK(S.scalar_type() == at::kLong && nnz.scalar_type() == at::kLong, "csr int64");
    TORCH_CHECK(k >= 0 && k <= kMaxFused, "k out of fused range");
    const int p = static_cast<int>(G.size(0));
    const int max_d = nbrs.dim() == 2 ? static_cast<int>(nbrs.size(1)) : 0;
    TORCH_CHECK(p > 0 && max_d > 0, "empty graph");
    const c10::cuda::CUDAGuard device_guard(G.device());
    const int64_t cap = static_cast<int64_t>(p) * max_d;
    auto dev = G.device();
    auto i32 = torch::TensorOptions().dtype(at::kInt).device(dev);
    auto i64 = torch::TensorOptions().dtype(at::kLong).device(dev);
    auto u8 = torch::TensorOptions().dtype(at::kByte).device(dev);
    const int64_t RANK_MAX = std::numeric_limits<int64_t>::max();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    unsigned long long* mptr = nullptr;
    if (fci && min_abs.defined() && min_abs.numel() > 0) {
        TORCH_CHECK(min_abs.is_cuda() && min_abs.scalar_type() == at::kLong, "min_abs int64 CUDA");
        TORCH_CHECK(min_abs.numel() >= cap, "min_abs capacity");
        mptr = reinterpret_cast<unsigned long long*>(min_abs.data_ptr<int64_t>());
    }
    auto* pmax_ptr = fci && pmax.defined() && pmax.numel() > 0 ? pmax.data_ptr<double>() : nullptr;

    const bool pair_cfg = k <= 3 && pc_smallk_enabled();
    // Hub fallback. Pair-major parallelism is capped by the number of live
    // pairs, not by the work per pair: a hub node makes one CTA execute a huge
    // C(deg-1, k) range alone (k=3, deg~500 is ~2e7 tests in one 256-thread
    // block). When the worst-case pair exceeds ~512 block waves of that kernel,
    // use the flat kernel, whose grid is one thread per test and therefore
    // scales with the total work. This only selects a path; the CI math and
    // the resulting sepsets/G are identical, which the A/B tests pin.
    bool hub_pair = false;
    if (pair_cfg && k >= 1) {
        const int pair_threads = (k == 1) ? 32 : (k == 2 ? 128 : 256);
        const int64_t pair_tests_upper = binom_host(max_d - 1, static_cast<int>(k));
        hub_pair = pair_tests_upper > static_cast<int64_t>(pair_threads) * 512;
    }
    const bool use_pair_major = pair_cfg && !hub_pair;

    // Fast path: pair-major CI + cached per-device workspace. The depth needs
    // no offset scan and no pre-launch test total; only the live pair count is
    // copied back (to size the CI grid), then every remaining transfer is an
    // async copy into pinned staging followed by one stream sync at the end.
    if (use_pair_major && pc_workspace_enabled()) {
        PcDepthWorkspace& w = pc_depth_workspace(G.get_device());
        pc_depth_workspace_ensure(w, p, cap, k > 0, i32, i64, u8);
        int64_t* host = w.pinned.data_ptr<int64_t>();
        auto ret = torch::empty({3}, torch::TensorOptions().dtype(at::kLong));
        int64_t* out = ret.data_ptr<int64_t>();
        cudaMemsetAsync(w.E_out.data_ptr<int>(), 0, sizeof(int), stream);
        pc_live_pairs_kernel<<<grid(cap), kBlock, 0, stream>>>(
            nbrs.data_ptr<int>(), deg.data_ptr<int>(), p, max_d, static_cast<int>(k),
            w.px.data_ptr<int>(), w.py.data_ptr<int>(), w.ncomb.data_ptr<int64_t>(),
            w.E_out.data_ptr<int>(), w.first.data_ptr<int64_t>(), k == 0 ? 1 : 0,
            G.data_ptr<int8_t>(), G.stride(0), G.stride(1));
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        if (k >= 1) {
            sum_ncomb_live_kernel<<<1, kBlock, 0, stream>>>(w.ncomb.data_ptr<int64_t>(),
                                                            w.E_out.data_ptr<int>(),
                                                            w.n_tests.data_ptr<int64_t>());
            WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        }
        cudaMemcpyAsync(host, w.E_out.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(w.e_event, stream);
        cudaEventSynchronize(w.e_event);
        const int E_live = *reinterpret_cast<int*>(host);
        if (E_live <= 0) {
            cudaMemcpyAsync(&host[2], nnz.data_ptr<int64_t>(), sizeof(int64_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            out[0] = 0;
            out[1] = host[2];
            out[2] = 0;
            return ret;
        }
        cudaMemsetAsync(w.overflow.data_ptr<int>(), 0, sizeof(int), stream);
        if (mptr != nullptr) {
            cudaMemsetAsync(mptr, 0xFF, static_cast<size_t>(cap) * sizeof(unsigned long long),
                            stream);
        }
        if (k >= 1) {
            cudaMemsetAsync(w.union_marks.data_ptr<uint8_t>(), 0,
                            static_cast<size_t>(E_live) * static_cast<size_t>(p), stream);
            cudaMemsetAsync(w.hit_flags.data_ptr<uint8_t>(), 0, static_cast<size_t>(E_live),
                            stream);
            graph_batch_ci_from_comb_pair_major(C, G, nbrs, deg, w.px, w.py, w.empty_i64, w.ncomb,
                                                k, w.first, E_live, n_samples, tau, eps, false,
                                                torch::Tensor(), w.union_marks, torch::Tensor(),
                                                w.hit_flags, fci, min_abs);
            if (pmax_ptr != nullptr) {
                pmax_update_kernel<<<grid(E_live), kBlock, 0, stream>>>(
                    mptr, w.px.data_ptr<int>(), w.py.data_ptr<int>(), E_live, pmax_ptr, p,
                    static_cast<int>(n_samples), static_cast<int>(k), k == 0 ? 1 : 0);
                WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
            }
            const int64_t p2 = static_cast<int64_t>(p) * p;
            int* tab = w.tab.data_ptr<int>();
            // The cached workspace survives across depths and graphs. Reset the
            // pair table every depth so a key with no hit this depth cannot
            // expose a stale pair index from an earlier depth or case.
            cudaMemsetAsync(tab, 0xFF, static_cast<size_t>(2 * p2) * sizeof(int), stream);
            commit_union_mark_kernel<<<grid(E_live), kBlock, 0, stream>>>(
                w.px.data_ptr<int>(), w.py.data_ptr<int>(), w.hit_flags.data_ptr<uint8_t>(), E_live,
                p, tab, tab + p2);
            commit_union_write_fused_kernel<<<grid(p2), kBlock, 0, stream>>>(
                tab, tab + p2, w.union_marks.data_ptr<uint8_t>(), p, p, ranges.data_ptr<int64_t>(),
                nnz.data_ptr<int64_t>(), S.data_ptr<int64_t>(), static_cast<int64_t>(S.numel()),
                w.overflow.data_ptr<int>(), w.px.data_ptr<int>(), w.py.data_ptr<int>(),
                G.data_ptr<int8_t>(), G.stride(0), G.stride(1));
            WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            graph_batch_ci_from_comb_pair_major(C, G, nbrs, deg, w.px, w.py, w.empty_i64, w.ncomb,
                                                0, w.first, E_live, n_samples, tau, eps, false,
                                                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                                                torch::Tensor(), fci, min_abs);
            if (pmax_ptr != nullptr) {
                pmax_update_kernel<<<grid(E_live), kBlock, 0, stream>>>(
                    mptr, w.px.data_ptr<int>(), w.py.data_ptr<int>(), E_live, pmax_ptr, p,
                    static_cast<int>(n_samples), 0, 1);
                WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
            }
            claim_sepsets_kernel<<<grid(E_live), kBlock, 0, stream>>>(
                w.px.data_ptr<int>(), w.py.data_ptr<int>(), w.first.data_ptr<int64_t>(), nullptr, 0,
                RANK_MAX, nbrs.data_ptr<int>(), deg.data_ptr<int>(), E_live, p, max_d, 0,
                static_cast<int64_t>(S.numel()), ranges.data_ptr<int64_t>(),
                nnz.data_ptr<int64_t>(), S.data_ptr<int64_t>(), G.data_ptr<int8_t>(), G.stride(0),
                G.stride(1), 1, w.overflow.data_ptr<int>());
            WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (k >= 1) {
            cudaMemcpyAsync(&host[1], w.n_tests.data_ptr<int64_t>(), sizeof(int64_t),
                            cudaMemcpyDeviceToHost, stream);
        }
        cudaMemcpyAsync(&host[2], nnz.data_ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(reinterpret_cast<int*>(&host[3]), w.overflow.data_ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        out[0] = (k == 0) ? static_cast<int64_t>(E_live) : host[1];
        out[1] = host[2];
        out[2] = static_cast<int>(host[3]);
        return ret;
    }

    auto px = torch::empty({cap}, i32);
    auto py = torch::empty({cap}, i32);
    auto ncomb = torch::zeros({cap}, i64);
    auto off = torch::empty({cap + 1}, i64);
    auto E_out = torch::zeros({1}, i32);
    pc_live_pairs_kernel<<<grid(cap), kBlock, 0, stream>>>(
        nbrs.data_ptr<int>(), deg.data_ptr<int>(), p, max_d, static_cast<int>(k),
        px.data_ptr<int>(), py.data_ptr<int>(), ncomb.data_ptr<int64_t>(), E_out.data_ptr<int>(),
        nullptr, 0, nullptr, 0, 0);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    const int scan_threads = 1024;
    scan_counts_i64_kernel<<<1, scan_threads, scan_threads * sizeof(int64_t), stream>>>(
        ncomb.data_ptr<int64_t>(), off.data_ptr<int64_t>(), static_cast<int>(cap));
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t B = 0;
    int E_live = 0;
    cudaStreamSynchronize(stream);
    cudaMemcpy(&B, off.data_ptr<int64_t>() + cap, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&E_live, E_out.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    auto ret = torch::empty({3}, torch::TensorOptions().dtype(at::kLong));
    ret.data_ptr<int64_t>()[0] = B;
    if (B == 0) {
        cudaMemcpy(ret.data_ptr<int64_t>() + 1, nnz.data_ptr<int64_t>(), sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        ret.data_ptr<int64_t>()[2] = 0;
        return ret;
    }
    if (mptr != nullptr) {
        cudaMemsetAsync(mptr, 0xFF, static_cast<size_t>(cap) * sizeof(unsigned long long), stream);
    }
    auto first = torch::full({cap}, RANK_MAX, i64);
    auto empty64 = torch::empty({0}, i64);
    auto empty32 = torch::empty({0}, i32);
    auto empty8 = torch::empty({0}, u8);
    auto emptyf = torch::empty({0}, C.options());
    auto overflow = torch::zeros({1}, i32);
    if (k == 0) {
        if (use_pair_major) {
            graph_batch_ci_from_comb_pair_major(C, G, nbrs, deg, px, py, off, ncomb, 0, first,
                                                E_live, n_samples, tau, eps, false, emptyf, empty8,
                                                empty32, empty8, fci, min_abs);
        } else {
            graph_batch_ci_from_comb(C, G, nbrs, deg, px, py, off, 0, 0, empty64, first, B,
                                     n_samples, tau, eps, fci, false, emptyf, empty8, empty8,
                                     empty32, empty8, min_abs);
        }
        if (pmax_ptr != nullptr) {
            pmax_update_kernel<<<grid(E_live), kBlock, 0, stream>>>(
                mptr, px.data_ptr<int>(), py.data_ptr<int>(), E_live, pmax_ptr, p,
                static_cast<int>(n_samples), 0, 1);
            WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        }
        graph_batch_claim_sepsets(px, py, first, empty64, 0, RANK_MAX, nbrs, deg, ranges, nnz, S,
                                  overflow, 0);
        zero_claimed_kernel<<<grid(static_cast<int64_t>(p) * p), kBlock, 0, stream>>>(
            G.data_ptr<int8_t>(), G.stride(0), G.stride(1), ranges.data_ptr<int64_t>(), p);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        auto union_marks = torch::zeros({cap, static_cast<int64_t>(p)}, u8);
        auto hit_flags = torch::zeros({cap}, u8);
        if (use_pair_major) {
            graph_batch_ci_from_comb_pair_major(C, G, nbrs, deg, px, py, off, ncomb, k, first,
                                                E_live, n_samples, tau, eps, false, emptyf,
                                                union_marks, empty32, hit_flags, fci, min_abs);
        } else {
            graph_batch_ci_from_comb(C, G, nbrs, deg, px, py, off, static_cast<int>(k), 0, empty64,
                                     first, B, n_samples, tau, eps, fci, false, emptyf, empty8,
                                     union_marks, empty32, hit_flags, min_abs);
        }
        if (pmax_ptr != nullptr) {
            pmax_update_kernel<<<grid(E_live), kBlock, 0, stream>>>(
                mptr, px.data_ptr<int>(), py.data_ptr<int>(), E_live, pmax_ptr, p,
                static_cast<int>(n_samples), static_cast<int>(k), 0);
            WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        }
        auto tab = torch::full({2 * static_cast<int64_t>(p) * p}, -1, i32);
        auto tab_a = tab.narrow(0, 0, static_cast<int64_t>(p) * p);
        auto tab_b = tab.narrow(0, static_cast<int64_t>(p) * p, static_cast<int64_t>(p) * p);
        graph_batch_commit_union_mark(px, py, hit_flags, tab_a, tab_b, p);
        graph_batch_commit_union_write(tab_a, tab_b, union_marks, ranges, nnz, S, overflow);
        graph_batch_zero_flagged(G, px, py, hit_flags);
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(ret.data_ptr<int64_t>() + 1, nnz.data_ptr<int64_t>(), sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    int32_t overflow_host = 0;
    cudaMemcpy(&overflow_host, overflow.data_ptr<int>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ret.data_ptr<int64_t>()[2] = overflow_host;
    return ret;
}

// Full PC-stable union skeleton driven from C++: the depth loop, per-depth
// adjacency compaction, the fused CI, and the sepset commit all stay in the
// extension. Python issues ONE call and reads back a small CPU tensor.
//
// Returns ``{nnz_after, fallback_depth, final_depth, n_tests, overflow}``:
//   * ``fallback_depth`` is ``-1`` on a clean finish, else the first depth
//     past ``k_max_fused`` (the caller continues with the host wavefront);
//   * ``final_depth`` is the last depth processed;
//   * ``overflow`` counts sepset-append reservations that did not fit in ``S``.
torch::Tensor graph_batch_pc_depth_union(torch::Tensor C, torch::Tensor G, torch::Tensor nbrs,
                                         torch::Tensor deg, torch::Tensor ranges, torch::Tensor S,
                                         torch::Tensor nnz, double tau, double eps,
                                         int64_t n_samples, int64_t k) {
    return graph_batch_pc_depth_union_fci(C, G, nbrs, deg, ranges, S, nnz, tau, eps, n_samples, k,
                                          false, torch::Tensor(), torch::Tensor());
}

torch::Tensor graph_batch_pc_skeleton_union(torch::Tensor C, torch::Tensor G, torch::Tensor ranges,
                                            torch::Tensor S, torch::Tensor nnz, double tau,
                                            double eps, int64_t n_samples, int64_t k_max_fused,
                                            int64_t max_depth, bool fci, torch::Tensor pmax,
                                            torch::Tensor min_abs) {
    TORCH_CHECK(G.is_cuda() && G.scalar_type() == at::kChar && G.dim() == 2 &&
                    G.size(0) == G.size(1),
                "G must be a CUDA int8 square matrix");
    const int p = static_cast<int>(G.size(0));
    TORCH_CHECK(p > 0, "empty graph");
    auto dev = G.device();
    auto i32 = torch::TensorOptions().dtype(at::kInt).device(dev);
    auto i64 = torch::TensorOptions().dtype(at::kLong).device(dev);
    auto ret = torch::zeros({5}, torch::TensorOptions().dtype(at::kLong));
    int64_t* out = ret.data_ptr<int64_t>();
    // A top-level skeleton call owns the depth workspace for its lifetime.
    // Force fresh pair/union buffers so state from an earlier call in the same
    // process (possibly with a different graph) can never leak into this one.
    {
        auto& ws = pc_depth_workspace(G.get_device());
        ws.p = -1;
        ws.cap = 0;
    }
    int depth = -1;
    int64_t total_tests = 0;
    int fallback = -1;
    int overflow_total = 0;
    while (true) {
        auto deg = torch::zeros({p}, i32);
        graph_batch_degrees(G, deg);
        const int max_d_cur = std::max(1, deg.max().item<int>());
        if (max_d_cur - 1 <= depth) {
            break;
        }
        if (max_depth >= 0 && depth + 1 > max_depth) {
            break;
        }
        depth += 1;
        if (depth > k_max_fused) {
            fallback = depth;
            depth -= 1; // not processed
            break;
        }
        auto nbrs = torch::zeros({p, max_d_cur}, i32);
        graph_batch_compact(G, nbrs, deg);
        auto vals = graph_batch_pc_depth_union_fci(C, G, nbrs, deg, ranges, S, nnz, tau, eps,
                                                   n_samples, depth, fci, pmax, min_abs);
        const int64_t* host = vals.data_ptr<int64_t>();
        total_tests += host[0];
        overflow_total += static_cast<int>(host[2]);
    }
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    out[0] = nnz.item<int64_t>();
    out[1] = fallback;
    out[2] = depth;
    out[3] = total_tests;
    out[4] = overflow_total;
    return ret;
}

void graph_batch_preload() {
    cuda_preload_context();
    touch_graph_batch_anon();
#if WHYDRA_ENABLE_FLOAT32
    touch_cuda_kernel(prefix_pmax_kernel<float>);
#endif
    touch_cuda_kernel(prefix_pmax_kernel<double>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("degrees", &graph_batch_degrees);
    m.def("compact", &graph_batch_compact);
    m.def("emit_pairs", &graph_batch_emit_pairs);
    m.def("count_tests", &graph_batch_count_tests);
    m.def("fill_batch", &graph_batch_fill);
    m.def("ci_from_comb", &graph_batch_ci_from_comb);
    m.def("prefix_pmax", &graph_batch_prefix_pmax);
    m.def("first_hit", &graph_batch_first_hit);
    m.def("fill_at_ranks", &graph_batch_fill_at_ranks);
    m.def("claim_sepsets", &graph_batch_claim_sepsets);
    m.def("pc_live_pairs", &graph_batch_pc_live_pairs);
    m.def("zero_flagged", &graph_batch_zero_flagged);
    m.def("pc_depth_union", &graph_batch_pc_depth_union);
    m.def("pc_depth_union_fci", &graph_batch_pc_depth_union_fci, pybind11::arg("C"),
          pybind11::arg("G"), pybind11::arg("nbrs"), pybind11::arg("deg"), pybind11::arg("ranges"),
          pybind11::arg("S"), pybind11::arg("nnz"), pybind11::arg("tau"), pybind11::arg("eps"),
          pybind11::arg("n_samples"), pybind11::arg("k"), pybind11::arg("fci"),
          pybind11::arg("pmax"), pybind11::arg("min_abs"));
    m.def("pc_skeleton_union", &graph_batch_pc_skeleton_union, pybind11::arg("C"),
          pybind11::arg("G"), pybind11::arg("ranges"), pybind11::arg("S"), pybind11::arg("nnz"),
          pybind11::arg("tau"), pybind11::arg("eps"), pybind11::arg("n_samples"),
          pybind11::arg("k_max_fused"), pybind11::arg("max_depth"), pybind11::arg("fci") = false,
          pybind11::arg("pmax") = torch::Tensor(), pybind11::arg("min_abs") = torch::Tensor());
    m.def("commit_union_mark", &graph_batch_commit_union_mark);
    m.def("commit_union_write", &graph_batch_commit_union_write);
    m.def("pdsep_commit_walk", &graph_batch_pdsep_commit_walk);
    m.def("hits_from_r", &graph_batch_hits_from_r);
    m.def("apply_hits", &graph_batch_apply_hits);
    m.def("edge_present", &graph_batch_edge_present);
    m.def("delete_if_indep", &graph_batch_delete_if_indep);
    m.def("pack_unique", &graph_batch_pack_unique);
    m.def("pack_ncomb", &graph_batch_pack_ncomb);
    m.def("compact_live", &graph_batch_compact_live);
    m.def("mark_s_k1", &graph_batch_mark_s_k1);
    m.def("mark_s_k2", &graph_batch_mark_s_k2);
    m.def("mark_s_ranks", &graph_batch_mark_s_ranks);
    m.def("unrank_rows", &graph_batch_unrank_rows);
    m.def("count_s_tests", &graph_batch_count_s_tests);
    m.def("fill_s_tests", &graph_batch_fill_s_tests);
    m.def("count_nbr_combos", &graph_batch_count_nbr_combos);
    m.def("fill_nbr_combos", &graph_batch_fill_nbr_combos);
    m.def("k_max", &graph_batch_k_max);
    m.def("fused_k_max", &graph_batch_fused_k_max);
    m.def("preload", &graph_batch_preload);
}
