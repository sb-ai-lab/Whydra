// Pearson χ²: one CTA per test, block 256. Recoded codes in column-major
// uint16 data[var * n + t], values in 0 .. c_j-1, c_j ≤ kCFull (1024).
//
// V×C SMEM strips, V ∈ {4,8,16,32}. Host `tile_v_for_card(C)` picks V.
// k=0 walks each sample once via X CSR. Y is staged into SMEM when it
// fits. After each strip, Q_b=Σ_a O²/r_a and c_b accumulate.
// χ² = n Σ Q_b/c_b − n. k=0 df = (c_x−1)(c_y−1), no empty-row drop.
// k=1: each Z-stratum is that 1-phase kernel on the slice (drop empty
// rows/cols). k≥2 groups samples by pack(Z) with CSR counting-sort
// (not comparison sort): if C^k ≤ kCFull, one packed CSR and the k=1
// loop; otherwise LSD counting-sort by each Z column, then Pearson on
// occupied runs (≤ n strata, not C^k).
//
// The k≥2 index buffers live in SMEM when they fit; otherwise the same
// test body runs with them in a global-memory workspace (two int arrays
// per persistent CTA, grid-stride over the batch).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_check.cuh"
#include <climits>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_touch.cuh"

namespace {

constexpr int kBlock = 256;
constexpr int kCFull = 1024;
// Q/col/X-CSR heads are static `__shared__` of kCFull (~21 KB).
constexpr int kStaticReserve = 32768;
using Code = uint16_t;

inline int optin_dyn_smem();
int tile_v_for_card(int C);
int resolve_tile_v(int C, int requested);

template <typename Kernel> void enable_dyn_smem(Kernel kernel, size_t smem) {
    const int cap = optin_dyn_smem();
    const int dyn_cap = cap - kStaticReserve;
    TORCH_CHECK(dyn_cap > 0, "static smem exceeds device opt-in");
    TORCH_CHECK(static_cast<int>(smem) <= dyn_cap, "χ² dynamic smem ", smem,
                " exceeds opt-in minus static reserve ", dyn_cap);
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     dyn_cap) == cudaSuccess,
                "cudaFuncSetAttribute max dynamic smem");
}

inline size_t strip_hist_bytes(int V, int C) {
    return sizeof(int) * static_cast<size_t>(V) * static_cast<size_t>(C);
}

inline int dyn_smem_cap() { return optin_dyn_smem() - kStaticReserve; }

inline size_t k0_tiled_smem(int V, int C, int n, int cap, int* cache_y) {
    const size_t hist = strip_hist_bytes(V, C);
    const size_t yb = sizeof(Code) * static_cast<size_t>(n);
    const size_t want = hist + yb;
    if (n > 0 && want <= static_cast<size_t>(cap)) {
        *cache_y = 1;
        return want;
    }
    *cache_y = 0;
    return hist;
}

inline size_t k1_tiled_smem(int V, int C, int n, int cap, int* cache_xy, int* local_csr) {
    const size_t hist = strip_hist_bytes(V, C);
    const size_t xy = 2 * sizeof(Code) * static_cast<size_t>(n);
    const size_t loc = sizeof(int) * static_cast<size_t>(n);
    const size_t pad = 4;
    const size_t full = hist + xy + pad + loc;
    if (n > 0 && full <= static_cast<size_t>(cap)) {
        *cache_xy = 1;
        *local_csr = 1;
        return full;
    }
    const size_t loc_only = hist + pad + loc;
    if (n > 0 && loc_only <= static_cast<size_t>(cap)) {
        *cache_xy = 0;
        *local_csr = 1;
        return loc_only;
    }
    const size_t xy_only = hist + xy;
    if (n > 0 && xy_only <= static_cast<size_t>(cap)) {
        *cache_xy = 1;
        *local_csr = 0;
        return xy_only;
    }
    *cache_xy = 0;
    *local_csr = 0;
    return hist;
}

inline size_t k1_gmem_smem(int V, int C, int n, int cap, int* cache_xy) {
    const size_t hist = strip_hist_bytes(V, C);
    const size_t xy = 2 * sizeof(Code) * static_cast<size_t>(n);
    *cache_xy = 0;
    if (n > 0 && hist + xy <= static_cast<size_t>(cap)) {
        *cache_xy = 1;
        return hist + xy;
    }
    return hist;
}

inline size_t k_pack_tiled_smem(int V, int C, int n, int cap, int* cache_xy) {
    const size_t hist = strip_hist_bytes(V, C);
    const size_t xy = 2 * sizeof(Code) * static_cast<size_t>(n);
    const size_t two_idx = 2 * sizeof(int) * static_cast<size_t>(n);
    const size_t pad = 16;
    *cache_xy = 0;
    if (n <= 0) {
        return hist;
    }
    const size_t full = hist + xy + pad + two_idx;
    if (full <= static_cast<size_t>(cap)) {
        *cache_xy = 1;
        return full;
    }
    const size_t idx_only = hist + pad + two_idx;
    if (idx_only <= static_cast<size_t>(cap)) {
        return idx_only;
    }
    return 0;
}

inline size_t k_pack_gmem_smem(int V, int C, int n, int cap, int* cache_xy) {
    const size_t hist = strip_hist_bytes(V, C);
    const size_t pad = 16;
    *cache_xy = 0;
    if (n <= 0) {
        return hist;
    }
    const size_t full = hist + 2 * sizeof(Code) * static_cast<size_t>(n) + pad;
    if (full <= static_cast<size_t>(cap)) {
        *cache_xy = 1;
        return full;
    }
    return hist;
}

inline bool valid_tile_v(int V) { return V == 4 || V == 8 || V == 16 || V == 32; }

// Fastest V that still fits the opt-in SMEM budget on the target GPU.
// (n=4096, p=16, A40). k=1 dominates PC/FCI time, so the table follows k=1;
// k=0 is within ~20% of its own best at the same V.
int tile_v_for_card(int C) {
    if (C < 1) {
        C = 1;
    }
    if (C > kCFull) {
        C = kCFull;
    }
    int prefer;
    if (C <= 512) {
        prefer = 32;
    } else {
        prefer = 16;
    }
    const int cap = dyn_smem_cap();
    for (int V = prefer; V >= 4; V >>= 1) {
        if (strip_hist_bytes(V, C) <= static_cast<size_t>(cap) && valid_tile_v(V)) {
            return V;
        }
    }
    return 4;
}

int resolve_tile_v(int C, int requested) {
    if (valid_tile_v(requested)) {
        return requested;
    }
    return tile_v_for_card(C);
}

__device__ void zero_ints(int* p, int n, int tid) {
    for (int i = tid; i < n; i += kBlock) {
        p[i] = 0;
    }
}

__device__ void load_codes(const Code* src, Code* dst, int n, int tid) {
    for (int t = tid; t < n; t += kBlock) {
        dst[t] = src[t];
    }
}

__device__ void hist_strip_from_x_csr(const int* idx_x, const int* off_x, const Code* Ycol, int* o,
                                      int a0, int a_end, int C, int tid) {
    for (int a = a0; a < a_end; ++a) {
        const int s = off_x[a];
        const int e = off_x[a + 1];
        for (int k = s + tid; k < e; k += kBlock) {
            const int t = idx_x[k];
            atomicAdd(&o[(a - a0) * C + static_cast<int>(Ycol[t])], 1);
        }
    }
}

__device__ void exclusive_prefix(int* off, const int* cnt, int n, int tid) {
    if (tid == 0) {
        int s = 0;
        off[0] = 0;
        for (int i = 0; i < n; ++i) {
            s += cnt[i];
            off[i + 1] = s;
        }
    }
}

__device__ void build_x_csr_slice(const int* idx_src, int zs, int ze, const Code* Xcol, int* off,
                                  int* head, int* idx_loc, int cx, int tid) {
    zero_ints(head, cx, tid);
    __syncthreads();
    for (int k = zs + tid; k < ze; k += kBlock) {
        const int t = idx_src[k];
        atomicAdd(&head[static_cast<int>(Xcol[t])], 1);
    }
    __syncthreads();
    exclusive_prefix(off, head, cx, tid);
    __syncthreads();
    for (int a = tid; a < cx; a += kBlock) {
        head[a] = off[a];
    }
    __syncthreads();
    for (int k = zs + tid; k < ze; k += kBlock) {
        const int t = idx_src[k];
        const int a = static_cast<int>(Xcol[t]);
        const int p = atomicAdd(&head[a], 1);
        idx_loc[p] = t;
    }
    __syncthreads();
}

__device__ int pack_product_or_zero(const int* cards, const int* S, int k) {
    int64_t prod = 1;
    for (int i = 0; i < k; ++i) {
        const int64_t c = cards[S[i]];
        if (c < 1) {
            return 0;
        }
        if (prod > static_cast<int64_t>(kCFull) / c) {
            return 0;
        }
        prod *= c;
    }
    return static_cast<int>(prod);
}

__device__ int pack_S_int(const Code* data, const int* cards, const int* S, int k, int n, int t) {
    int z = 0;
    for (int i = 0; i < k; ++i) {
        const int v = S[i];
        z = z * cards[v] + static_cast<int>(data[static_cast<int64_t>(v) * n + t]);
    }
    return z;
}

__device__ bool z_tuple_eq(int t1, int t2, const Code* data, const int* S, int k, int n) {
    for (int i = 0; i < k; ++i) {
        const int v = S[i];
        if (data[static_cast<int64_t>(v) * n + t1] != data[static_cast<int64_t>(v) * n + t2]) {
            return false;
        }
    }
    return true;
}

__device__ void counting_sort_by_code(const int* src, int* dst, int n, const Code* Zcol, int cz,
                                      int* off, int* head, int tid) {
    zero_ints(head, cz, tid);
    __syncthreads();
    for (int t = tid; t < n; t += kBlock) {
        atomicAdd(&head[static_cast<int>(Zcol[src[t]])], 1);
    }
    __syncthreads();
    exclusive_prefix(off, head, cz, tid);
    __syncthreads();
    for (int a = tid; a < cz; a += kBlock) {
        head[a] = off[a];
    }
    __syncthreads();
    if (tid == 0) {
        for (int t = 0; t < n; ++t) {
            const int i = src[t];
            const int a = static_cast<int>(Zcol[i]);
            dst[head[a]++] = i;
        }
    }
    __syncthreads();
}

__device__ void counting_sort_packed(int* dst, int n, const Code* data, const int* cards,
                                     const int* S, int k, int cz, int* off, int* head, int tid) {
    zero_ints(head, cz, tid);
    __syncthreads();
    for (int t = tid; t < n; t += kBlock) {
        atomicAdd(&head[pack_S_int(data, cards, S, k, n, t)], 1);
    }
    __syncthreads();
    exclusive_prefix(off, head, cz, tid);
    __syncthreads();
    for (int a = tid; a < cz; a += kBlock) {
        head[a] = off[a];
    }
    __syncthreads();
    for (int t = tid; t < n; t += kBlock) {
        const int p = atomicAdd(&head[pack_S_int(data, cards, S, k, n, t)], 1);
        dst[p] = t;
    }
    __syncthreads();
}

__device__ void zero_q_col(double* Q, int* col, int n, int tid) {
    for (int i = tid; i < n; i += kBlock) {
        Q[i] = 0.0;
        col[i] = 0;
    }
}

// χ² = n Σ_b Q_b/c_b − n with Q_b = Σ_a O_ab² / r_a.
// A full-width V×C strip has exact r_a. Q and c accumulate across strips.
template <int V>
__device__ void accumulate_q_col(const int* o, double* Q, int* col, int* rloc, int* nr, int a0,
                                 int a_end, int cy, int C, int tid) {
    const int n_a = a_end - a0;
    if (tid < V) {
        int s = 0;
        if (tid < n_a) {
            const int* rowp = o + tid * C;
            for (int b = 0; b < cy; ++b) {
                s += rowp[b];
            }
        }
        rloc[tid] = s;
    }
    __syncthreads();
    if (tid == 0) {
        int npos = 0;
        for (int ia = 0; ia < n_a; ++ia) {
            if (rloc[ia] > 0) {
                ++npos;
            }
        }
        *nr += npos;
    }
    for (int b = tid; b < cy; b += kBlock) {
        int cs = 0;
        double q = 0.0;
        for (int ia = 0; ia < n_a; ++ia) {
            const int v = o[ia * C + b];
            cs += v;
            const int r = rloc[ia];
            if (r > 0 && v != 0) {
                q += static_cast<double>(v) * static_cast<double>(v) / static_cast<double>(r);
            }
        }
        col[b] += cs;
        Q[b] += q;
    }
}

__device__ void finish_pearson(const double* Q, const int* col, int cy, double& stat, int& n_sub,
                               int& nc) {
    int n_i = 0;
    double s = 0.0;
    nc = 0;
    for (int b = 0; b < cy; ++b) {
        const int cb = col[b];
        if (cb > 0) {
            n_i += cb;
            s += Q[b] / static_cast<double>(cb);
            ++nc;
        }
    }
    n_sub = n_i;
    const double n = static_cast<double>(n_i);
    stat = n > 0.0 ? n * s - n : 0.0;
}

template <int V>
__device__ void pearson_run(const int* idx_src, int zs, int ze, const Code* Xcol, const Code* Ycol,
                            int cx, int cy, int C, int* o, int* loc_off, int* loc_head,
                            int* idx_loc, double* Q, int* col, int* rloc, int* sh_nr,
                            double* sh_stat, int* sh_df, int tid) {
    if (ze - zs < 2) {
        return;
    }
    zero_q_col(Q, col, cy, tid);
    if (tid == 0) {
        *sh_nr = 0;
    }
    __syncthreads();
    build_x_csr_slice(idx_src, zs, ze, Xcol, loc_off, loc_head, idx_loc, cx, tid);
    for (int a0s = 0; a0s < cx; a0s += V) {
        const int a_end = a0s + V < cx ? a0s + V : cx;
        zero_ints(o, V * C, tid);
        __syncthreads();
        hist_strip_from_x_csr(idx_loc, loc_off, Ycol, o, a0s, a_end, C, tid);
        __syncthreads();
        accumulate_q_col<V>(o, Q, col, rloc, sh_nr, a0s, a_end, cy, C, tid);
        __syncthreads();
    }
    if (tid == 0) {
        double s = 0.0;
        int n_sub = 0;
        int nc = 0;
        finish_pearson(Q, col, cy, s, n_sub, nc);
        if (n_sub >= 2 && *sh_nr >= 2 && nc >= 2) {
            *sh_stat += s;
            *sh_df += (*sh_nr - 1) * (nc - 1);
        }
    }
    __syncthreads();
}

template <int V>
__global__ void chisq_k0_tiled_kernel(const Code* __restrict__ data, const int* __restrict__ cards,
                                      const int* __restrict__ x, const int* __restrict__ y,
                                      double* __restrict__ stat, int* __restrict__ df,
                                      const int* __restrict__ csr_off,
                                      const int* __restrict__ csr_idx, int off_stride, int C, int n,
                                      int cache_y, int64_t B) {
    const int64_t b = static_cast<int64_t>(blockIdx.x);
    if (b >= B) {
        return;
    }
    const int xv = x[b];
    const int yv = y[b];
    const int cx = cards[xv];
    const int cy = cards[yv];
    const int tid = threadIdx.x;
    extern __shared__ char raw[];
    int* o = reinterpret_cast<int*>(raw);
    Code* Ys = reinterpret_cast<Code*>(o + V * C);

    __shared__ double Q[kCFull];
    __shared__ int col[kCFull];
    __shared__ int rloc[V];
    __shared__ int sh_nr;

    if (cx > C || cy > C || cx < 1 || cy < 1 || n < 1) {
        if (tid == 0) {
            stat[b] = 0.0;
            df[b] = -1;
        }
        return;
    }

    const Code* Ycol = data + static_cast<int64_t>(yv) * n;
    if (cache_y) {
        load_codes(Ycol, Ys, n, tid);
        Ycol = Ys;
    }
    const int* off_x = csr_off + static_cast<int64_t>(xv) * off_stride;
    const int* idx_x = csr_idx + static_cast<int64_t>(xv) * n;
    zero_q_col(Q, col, cy, tid);
    if (tid == 0) {
        sh_nr = 0;
    }
    __syncthreads();

    for (int a0 = 0; a0 < cx; a0 += V) {
        const int a_end = a0 + V < cx ? a0 + V : cx;
        zero_ints(o, V * C, tid);
        __syncthreads();
        hist_strip_from_x_csr(idx_x, off_x, Ycol, o, a0, a_end, C, tid);
        __syncthreads();
        accumulate_q_col<V>(o, Q, col, rloc, &sh_nr, a0, a_end, cy, C, tid);
        __syncthreads();
    }
    if (tid == 0) {
        double s = 0.0;
        int n_sub = 0;
        int nc = 0;
        finish_pearson(Q, col, cy, s, n_sub, nc);
        stat[b] = s;
        df[b] = (cx - 1) * (cy - 1);
    }
}

template <int V>
__global__ void chisq_k1_tiled_kernel(const Code* __restrict__ data, const int* __restrict__ cards,
                                      const int* __restrict__ x, const int* __restrict__ y,
                                      const int* __restrict__ z, double* __restrict__ stat,
                                      int* __restrict__ df, const int* __restrict__ csr_off,
                                      const int* __restrict__ csr_idx, int off_stride, int C, int n,
                                      int cache_xy, int local_csr, int64_t B) {
    const int64_t b = static_cast<int64_t>(blockIdx.x);
    if (b >= B) {
        return;
    }
    const int xv = x[b];
    const int yv = y[b];
    const int zv = z[b];
    const int cx = cards[xv];
    const int cy = cards[yv];
    const int cz = cards[zv];
    const int tid = threadIdx.x;
    extern __shared__ char raw[];
    int* o = reinterpret_cast<int*>(raw);
    Code* Xs = reinterpret_cast<Code*>(o + V * C);
    Code* Ys = cache_xy ? Xs + n : Xs;
    int* idx_loc = nullptr;
    if (local_csr) {
        Code* after = cache_xy ? Ys + n : reinterpret_cast<Code*>(o + V * C);
        const size_t aligned = (reinterpret_cast<size_t>(after) + 3u) & ~static_cast<size_t>(3u);
        idx_loc = reinterpret_cast<int*>(aligned);
    }

    __shared__ double Q[kCFull];
    __shared__ int col[kCFull];
    __shared__ int rloc[V];
    __shared__ int loc_off[kCFull + 1];
    __shared__ int loc_head[kCFull];
    __shared__ int sh_nr;
    __shared__ double sh_stat;
    __shared__ int sh_df;

    if (cx > C || cy > C || cz > kCFull || cx < 1 || cy < 1 || cz < 1 || n < 1) {
        if (tid == 0) {
            stat[b] = 0.0;
            df[b] = -1;
        }
        return;
    }

    const Code* Xcol = data + static_cast<int64_t>(xv) * n;
    const Code* Ycol = data + static_cast<int64_t>(yv) * n;
    if (cache_xy) {
        for (int t = tid; t < n; t += kBlock) {
            Xs[t] = Xcol[t];
            Ys[t] = Ycol[t];
        }
        Xcol = Xs;
        Ycol = Ys;
    }
    const int* off_z = csr_off + static_cast<int64_t>(zv) * off_stride;
    const int* idx_z = csr_idx + static_cast<int64_t>(zv) * n;

    if (tid == 0) {
        sh_stat = 0.0;
        sh_df = 0;
    }
    __syncthreads();

    for (int zi = 0; zi < cz; ++zi) {
        const int zs = off_z[zi];
        const int ze = off_z[zi + 1];
        if (ze - zs < 2) {
            continue;
        }
        zero_q_col(Q, col, cy, tid);
        if (tid == 0) {
            sh_nr = 0;
        }
        __syncthreads();
        if (local_csr) {
            build_x_csr_slice(idx_z, zs, ze, Xcol, loc_off, loc_head, idx_loc, cx, tid);
            for (int a0 = 0; a0 < cx; a0 += V) {
                const int a_end = a0 + V < cx ? a0 + V : cx;
                zero_ints(o, V * C, tid);
                __syncthreads();
                hist_strip_from_x_csr(idx_loc, loc_off, Ycol, o, a0, a_end, C, tid);
                __syncthreads();
                accumulate_q_col<V>(o, Q, col, rloc, &sh_nr, a0, a_end, cy, C, tid);
                __syncthreads();
            }
        } else {
            for (int a0 = 0; a0 < cx; a0 += V) {
                const int a_end = a0 + V < cx ? a0 + V : cx;
                zero_ints(o, V * C, tid);
                __syncthreads();
                for (int k = zs + tid; k < ze; k += kBlock) {
                    const int t = idx_z[k];
                    const int a = static_cast<int>(Xcol[t]);
                    if (a < a0 || a >= a_end) {
                        continue;
                    }
                    atomicAdd(&o[(a - a0) * C + static_cast<int>(Ycol[t])], 1);
                }
                __syncthreads();
                accumulate_q_col<V>(o, Q, col, rloc, &sh_nr, a0, a_end, cy, C, tid);
                __syncthreads();
            }
        }
        if (tid == 0) {
            double s = 0.0;
            int n_sub = 0;
            int nc = 0;
            finish_pearson(Q, col, cy, s, n_sub, nc);
            if (n_sub >= 2 && sh_nr >= 2 && nc >= 2) {
                sh_stat += s;
                sh_df += (sh_nr - 1) * (nc - 1);
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        stat[b] = sh_stat;
        df[b] = sh_df;
    }
}

// k=1 with the local X-CSR index buffer in global memory. The X/Y code
// columns stay in SMEM; only the per-CTA scatter buffer moves out.
template <int V>
__global__ void chisq_k1_gmem_kernel(const Code* __restrict__ data, const int* __restrict__ cards,
                                     const int* __restrict__ x, const int* __restrict__ y,
                                     const int* __restrict__ z, double* __restrict__ stat,
                                     int* __restrict__ df, const int* __restrict__ csr_off,
                                     const int* __restrict__ csr_idx, int off_stride, int C, int n,
                                     int cache_xy, int64_t B, int* __restrict__ ws_loc) {
    const int tid = threadIdx.x;
    extern __shared__ char raw[];
    int* o = reinterpret_cast<int*>(raw);
    Code* Xs = reinterpret_cast<Code*>(o + V * C);
    Code* Ys = cache_xy ? Xs + n : Xs;
    int* idx_loc = ws_loc + static_cast<int64_t>(blockIdx.x) * n;

    __shared__ double Q[kCFull];
    __shared__ int col[kCFull];
    __shared__ int rloc[V];
    __shared__ int loc_off[kCFull + 1];
    __shared__ int loc_head[kCFull];
    __shared__ int sh_nr;
    __shared__ double sh_stat;
    __shared__ int sh_df;

    for (int64_t b = static_cast<int64_t>(blockIdx.x); b < B;
         b += static_cast<int64_t>(gridDim.x)) {
        const int xv = x[b];
        const int yv = y[b];
        const int zv = z[b];
        const int cx = cards[xv];
        const int cy = cards[yv];
        const int cz = cards[zv];
        if (cx > C || cy > C || cz > kCFull || cx < 1 || cy < 1 || cz < 1 || n < 1) {
            if (tid == 0) {
                stat[b] = 0.0;
                df[b] = -1;
            }
            continue;
        }
        const Code* Xcol = data + static_cast<int64_t>(xv) * n;
        const Code* Ycol = data + static_cast<int64_t>(yv) * n;
        if (cache_xy) {
            for (int t = tid; t < n; t += kBlock) {
                Xs[t] = Xcol[t];
                Ys[t] = Ycol[t];
            }
            Xcol = Xs;
            Ycol = Ys;
        }
        const int* off_z = csr_off + static_cast<int64_t>(zv) * off_stride;
        const int* idx_z = csr_idx + static_cast<int64_t>(zv) * n;
        if (tid == 0) {
            sh_stat = 0.0;
            sh_df = 0;
        }
        __syncthreads();
        for (int zi = 0; zi < cz; ++zi) {
            const int zs = off_z[zi];
            const int ze = off_z[zi + 1];
            if (ze - zs < 2) {
                continue;
            }
            zero_q_col(Q, col, cy, tid);
            if (tid == 0) {
                sh_nr = 0;
            }
            __syncthreads();
            build_x_csr_slice(idx_z, zs, ze, Xcol, loc_off, loc_head, idx_loc, cx, tid);
            for (int a0 = 0; a0 < cx; a0 += V) {
                const int a_end = a0 + V < cx ? a0 + V : cx;
                zero_ints(o, V * C, tid);
                __syncthreads();
                hist_strip_from_x_csr(idx_loc, loc_off, Ycol, o, a0, a_end, C, tid);
                __syncthreads();
                accumulate_q_col<V>(o, Q, col, rloc, &sh_nr, a0, a_end, cy, C, tid);
                __syncthreads();
            }
            if (tid == 0) {
                double s = 0.0;
                int n_sub = 0;
                int nc = 0;
                finish_pearson(Q, col, cy, s, n_sub, nc);
                if (n_sub >= 2 && sh_nr >= 2 && nc >= 2) {
                    sh_stat += s;
                    sh_df += (sh_nr - 1) * (nc - 1);
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            stat[b] = sh_stat;
            df[b] = sh_df;
        }
        __syncthreads();
    }
}

template <int V>
__device__ void
chisq_pack_one(int64_t b, const Code* __restrict__ data, const int* __restrict__ cards,
               const int* __restrict__ x, const int* __restrict__ y, const int* __restrict__ S,
               double* __restrict__ stat, int* __restrict__ df, int C, int n, int k, int s_stride,
               int cache_xy, int* o, Code* Xs, Code* Ys, int* idx_z, int* idx_x, double* Q,
               int* col, int* rloc, int* loc_off, int* loc_head, int* pack_off, int* sh_nr,
               double* sh_stat, int* sh_df, int* sh_run, int* sh_cz, int tid) {
    const int xv = x[b];
    const int yv = y[b];
    const int cx = cards[xv];
    const int cy = cards[yv];
    const int* Sb = S + b * s_stride;

    if (cx > C || cy > C || cx < 1 || cy < 1 || k < 2 || n < 1) {
        if (tid == 0) {
            stat[b] = 0.0;
            df[b] = -1;
        }
        return;
    }

    const Code* Xcol = data + static_cast<int64_t>(xv) * n;
    const Code* Ycol = data + static_cast<int64_t>(yv) * n;
    if (cache_xy) {
        for (int t = tid; t < n; t += kBlock) {
            Xs[t] = Xcol[t];
            Ys[t] = Ycol[t];
        }
        Xcol = Xs;
        Ycol = Ys;
    }

    if (tid == 0) {
        *sh_cz = pack_product_or_zero(cards, Sb, k);
        *sh_stat = 0.0;
        *sh_df = 0;
    }
    __syncthreads();

    if (*sh_cz > 0) {
        counting_sort_packed(idx_z, n, data, cards, Sb, k, *sh_cz, pack_off, loc_head, tid);
        for (int zi = 0; zi < *sh_cz; ++zi) {
            pearson_run<V>(idx_z, pack_off[zi], pack_off[zi + 1], Xcol, Ycol, cx, cy, C, o, loc_off,
                           loc_head, idx_x, Q, col, rloc, sh_nr, sh_stat, sh_df, tid);
        }
    } else {
        for (int t = tid; t < n; t += kBlock) {
            idx_z[t] = t;
        }
        __syncthreads();
        int* src = idx_z;
        int* dst = idx_x;
        for (int j = k - 1; j >= 0; --j) {
            const int zj = Sb[j];
            const int cj = cards[zj];
            const Code* Zcol = data + static_cast<int64_t>(zj) * n;
            counting_sort_by_code(src, dst, n, Zcol, cj, loc_off, loc_head, tid);
            int* tmp = src;
            src = dst;
            dst = tmp;
        }
        if (src != idx_z) {
            for (int t = tid; t < n; t += kBlock) {
                idx_z[t] = src[t];
            }
            __syncthreads();
        }
        int i = 0;
        while (i < n) {
            if (tid == 0) {
                int j = i + 1;
                const int t0 = idx_z[i];
                while (j < n && z_tuple_eq(idx_z[j], t0, data, Sb, k, n)) {
                    ++j;
                }
                *sh_run = j;
            }
            __syncthreads();
            const int j = *sh_run;
            pearson_run<V>(idx_z, i, j, Xcol, Ycol, cx, cy, C, o, loc_off, loc_head, idx_x, Q, col,
                           rloc, sh_nr, sh_stat, sh_df, tid);
            i = j;
        }
    }
    if (tid == 0) {
        stat[b] = *sh_stat;
        df[b] = *sh_df;
    }
}

template <int V>
__global__ void chisq_k_pack_tiled_kernel(const Code* __restrict__ data,
                                          const int* __restrict__ cards, const int* __restrict__ x,
                                          const int* __restrict__ y, const int* __restrict__ S,
                                          double* __restrict__ stat, int* __restrict__ df, int C,
                                          int n, int k, int s_stride, int cache_xy, int64_t B) {
    const int64_t b = static_cast<int64_t>(blockIdx.x);
    if (b >= B) {
        return;
    }
    const int tid = threadIdx.x;
    extern __shared__ char raw[];
    int* o = reinterpret_cast<int*>(raw);
    Code* Xs = reinterpret_cast<Code*>(o + V * C);
    Code* Ys = cache_xy ? Xs + n : Xs;
    Code* after = cache_xy ? Ys + n : Xs;
    const size_t a0 = (reinterpret_cast<size_t>(after) + 3u) & ~static_cast<size_t>(3u);
    int* idx_z = reinterpret_cast<int*>(a0);
    int* idx_x = idx_z + n;

    __shared__ double Q[kCFull];
    __shared__ int col[kCFull];
    __shared__ int rloc[V];
    __shared__ int loc_off[kCFull + 1];
    __shared__ int loc_head[kCFull];
    __shared__ int pack_off[kCFull + 1];
    __shared__ int sh_nr;
    __shared__ double sh_stat;
    __shared__ int sh_df;
    __shared__ int sh_run;
    __shared__ int sh_cz;

    chisq_pack_one<V>(b, data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, o, Xs, Ys,
                      idx_z, idx_x, Q, col, rloc, loc_off, loc_head, pack_off, &sh_nr, &sh_stat,
                      &sh_df, &sh_run, &sh_cz, tid);
}

template <int V>
__global__ void chisq_k_pack_gmem_kernel(const Code* __restrict__ data,
                                         const int* __restrict__ cards, const int* __restrict__ x,
                                         const int* __restrict__ y, const int* __restrict__ S,
                                         double* __restrict__ stat, int* __restrict__ df, int C,
                                         int n, int k, int s_stride, int cache_xy, int64_t B,
                                         int* __restrict__ ws_z, int* __restrict__ ws_x) {
    const int tid = threadIdx.x;
    extern __shared__ char raw[];
    int* o = reinterpret_cast<int*>(raw);
    Code* Xs = reinterpret_cast<Code*>(o + V * C);
    Code* Ys = cache_xy ? Xs + n : Xs;
    int* idx_z = ws_z + static_cast<int64_t>(blockIdx.x) * n;
    int* idx_x = ws_x + static_cast<int64_t>(blockIdx.x) * n;

    __shared__ double Q[kCFull];
    __shared__ int col[kCFull];
    __shared__ int rloc[V];
    __shared__ int loc_off[kCFull + 1];
    __shared__ int loc_head[kCFull];
    __shared__ int pack_off[kCFull + 1];
    __shared__ int sh_nr;
    __shared__ double sh_stat;
    __shared__ int sh_df;
    __shared__ int sh_run;
    __shared__ int sh_cz;

    for (int64_t b = static_cast<int64_t>(blockIdx.x); b < B;
         b += static_cast<int64_t>(gridDim.x)) {
        chisq_pack_one<V>(b, data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, o, Xs, Ys,
                          idx_z, idx_x, Q, col, rloc, loc_off, loc_head, pack_off, &sh_nr, &sh_stat,
                          &sh_df, &sh_run, &sh_cz, tid);
        __syncthreads();
    }
}

template <int V>
void launch_k0_tiled(const Code* data, const int* cards, const int* x, const int* y, double* stat,
                     int* df, const int* csr_off, const int* csr_idx, int off_stride, int C, int n,
                     int cache_y, int64_t B, size_t smem, cudaStream_t stream) {
    enable_dyn_smem(chisq_k0_tiled_kernel<V>, smem);
    chisq_k0_tiled_kernel<V><<<static_cast<unsigned int>(B), kBlock, smem, stream>>>(
        data, cards, x, y, stat, df, csr_off, csr_idx, off_stride, C, n, cache_y, B);
}

void launch_k0_tiled_v(int V, const Code* data, const int* cards, const int* x, const int* y,
                       double* stat, int* df, const int* csr_off, const int* csr_idx,
                       int off_stride, int C, int n, int cache_y, int64_t B, size_t smem,
                       cudaStream_t stream) {
    switch (V) {
    case 4:
        launch_k0_tiled<4>(data, cards, x, y, stat, df, csr_off, csr_idx, off_stride, C, n, cache_y,
                           B, smem, stream);
        break;
    case 8:
        launch_k0_tiled<8>(data, cards, x, y, stat, df, csr_off, csr_idx, off_stride, C, n, cache_y,
                           B, smem, stream);
        break;
    case 16:
        launch_k0_tiled<16>(data, cards, x, y, stat, df, csr_off, csr_idx, off_stride, C, n,
                            cache_y, B, smem, stream);
        break;
    case 32:
        launch_k0_tiled<32>(data, cards, x, y, stat, df, csr_off, csr_idx, off_stride, C, n,
                            cache_y, B, smem, stream);
        break;
    default:
        TORCH_CHECK(false, "tile V must be 4, 8, 16, or 32");
    }
}

template <int V>
void launch_k1_tiled(const Code* data, const int* cards, const int* x, const int* y, const int* z,
                     double* stat, int* df, const int* csr_off, const int* csr_idx, int off_stride,
                     int C, int n, int cache_xy, int local_csr, int64_t B, size_t smem,
                     cudaStream_t stream) {
    enable_dyn_smem(chisq_k1_tiled_kernel<V>, smem);
    chisq_k1_tiled_kernel<V><<<static_cast<unsigned int>(B), kBlock, smem, stream>>>(
        data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n, cache_xy, local_csr, B);
}

void launch_k1_tiled_v(int V, const Code* data, const int* cards, const int* x, const int* y,
                       const int* z, double* stat, int* df, const int* csr_off, const int* csr_idx,
                       int off_stride, int C, int n, int cache_xy, int local_csr, int64_t B,
                       size_t smem, cudaStream_t stream) {
    switch (V) {
    case 4:
        launch_k1_tiled<4>(data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n,
                           cache_xy, local_csr, B, smem, stream);
        break;
    case 8:
        launch_k1_tiled<8>(data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n,
                           cache_xy, local_csr, B, smem, stream);
        break;
    case 16:
        launch_k1_tiled<16>(data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n,
                            cache_xy, local_csr, B, smem, stream);
        break;
    case 32:
        launch_k1_tiled<32>(data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n,
                            cache_xy, local_csr, B, smem, stream);
        break;
    default:
        TORCH_CHECK(false, "tile V must be 4, 8, 16, or 32");
    }
}

template <int V>
void launch_k1_gmem(const Code* data, const int* cards, const int* x, const int* y, const int* z,
                    double* stat, int* df, const int* csr_off, const int* csr_idx, int off_stride,
                    int C, int n, int cache_xy, int64_t B, int* ws_loc, int grid, size_t smem,
                    cudaStream_t stream) {
    enable_dyn_smem(chisq_k1_gmem_kernel<V>, smem);
    chisq_k1_gmem_kernel<V><<<grid, kBlock, smem, stream>>>(
        data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n, cache_xy, B, ws_loc);
}

void launch_k1_gmem_v(int V, const Code* data, const int* cards, const int* x, const int* y,
                      const int* z, double* stat, int* df, const int* csr_off, const int* csr_idx,
                      int off_stride, int C, int n, int cache_xy, int64_t B, int* ws_loc, int grid,
                      size_t smem, cudaStream_t stream) {
    switch (V) {
    case 4:
        launch_k1_gmem<4>(data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n,
                          cache_xy, B, ws_loc, grid, smem, stream);
        break;
    case 8:
        launch_k1_gmem<8>(data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n,
                          cache_xy, B, ws_loc, grid, smem, stream);
        break;
    case 16:
        launch_k1_gmem<16>(data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n,
                           cache_xy, B, ws_loc, grid, smem, stream);
        break;
    case 32:
        launch_k1_gmem<32>(data, cards, x, y, z, stat, df, csr_off, csr_idx, off_stride, C, n,
                           cache_xy, B, ws_loc, grid, smem, stream);
        break;
    default:
        TORCH_CHECK(false, "tile V must be 4, 8, 16, or 32");
    }
}

template <int V>
void launch_pack_tiled(const Code* data, const int* cards, const int* x, const int* y, const int* S,
                       double* stat, int* df, int C, int n, int k, int s_stride, int cache_xy,
                       int64_t B, size_t smem, cudaStream_t stream) {
    enable_dyn_smem(chisq_k_pack_tiled_kernel<V>, smem);
    chisq_k_pack_tiled_kernel<V><<<static_cast<unsigned int>(B), kBlock, smem, stream>>>(
        data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B);
}

void launch_pack_tiled_v(int V, const Code* data, const int* cards, const int* x, const int* y,
                         const int* S, double* stat, int* df, int C, int n, int k, int s_stride,
                         int cache_xy, int64_t B, size_t smem, cudaStream_t stream) {
    switch (V) {
    case 4:
        launch_pack_tiled<4>(data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, smem,
                             stream);
        break;
    case 8:
        launch_pack_tiled<8>(data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, smem,
                             stream);
        break;
    case 16:
        launch_pack_tiled<16>(data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, smem,
                              stream);
        break;
    case 32:
        launch_pack_tiled<32>(data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, smem,
                              stream);
        break;
    default:
        TORCH_CHECK(false, "tile V must be 4, 8, 16, or 32");
    }
}

template <int V>
void launch_pack_gmem(const Code* data, const int* cards, const int* x, const int* y, const int* S,
                      double* stat, int* df, int C, int n, int k, int s_stride, int cache_xy,
                      int64_t B, int* ws_z, int* ws_x, int grid, size_t smem, cudaStream_t stream) {
    enable_dyn_smem(chisq_k_pack_gmem_kernel<V>, smem);
    chisq_k_pack_gmem_kernel<V><<<grid, kBlock, smem, stream>>>(
        data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, ws_z, ws_x);
}

void launch_pack_gmem_v(int V, const Code* data, const int* cards, const int* x, const int* y,
                        const int* S, double* stat, int* df, int C, int n, int k, int s_stride,
                        int cache_xy, int64_t B, int* ws_z, int* ws_x, int grid, size_t smem,
                        cudaStream_t stream) {
    switch (V) {
    case 4:
        launch_pack_gmem<4>(data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, ws_z,
                            ws_x, grid, smem, stream);
        break;
    case 8:
        launch_pack_gmem<8>(data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, ws_z,
                            ws_x, grid, smem, stream);
        break;
    case 16:
        launch_pack_gmem<16>(data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, ws_z,
                             ws_x, grid, smem, stream);
        break;
    case 32:
        launch_pack_gmem<32>(data, cards, x, y, S, stat, df, C, n, k, s_stride, cache_xy, B, ws_z,
                             ws_x, grid, smem, stream);
        break;
    default:
        TORCH_CHECK(false, "tile V must be 4, 8, 16, or 32");
    }
}

void check_codes(const torch::Tensor& data) {
    TORCH_CHECK(data.is_cuda() && data.scalar_type() == torch::kUInt16, "data uint16 CUDA");
    TORCH_CHECK(data.dim() == 2 && data.is_contiguous(), "data must be contiguous (p, n)");
}

inline int optin_dyn_smem() {
    int dev = 0;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "cudaGetDevice");
    int n = 0;
    TORCH_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) ==
                    cudaSuccess,
                "cudaDevAttrMaxSharedMemoryPerBlockOptin");
    return n > 0 ? n : 48 * 1024;
}

void chisq_k0(torch::Tensor data, torch::Tensor cards, torch::Tensor x, torch::Tensor y,
              torch::Tensor stat, torch::Tensor df, int64_t max_card, torch::Tensor csr_off,
              torch::Tensor csr_idx, int64_t tile_v) {
    check_codes(data);
    TORCH_CHECK(cards.is_cuda() && cards.scalar_type() == torch::kInt32, "cards int32 CUDA");
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && stat.is_cuda() && df.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kInt32 && y.scalar_type() == torch::kInt32);
    TORCH_CHECK(stat.scalar_type() == torch::kFloat64 && df.scalar_type() == torch::kInt32);
    const int64_t B = x.numel();
    TORCH_CHECK(y.numel() == B && stat.numel() == B && df.numel() == B);
    const int n = static_cast<int>(data.size(1));
    if (B == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int requested = static_cast<int>(tile_v);
    TORCH_CHECK(max_card >= 1 && max_card <= kCFull, "max_card must be in 1..1024");
    const int C = static_cast<int>(max_card);
    const int V = resolve_tile_v(C, requested);
    TORCH_CHECK(csr_off.is_cuda() && csr_idx.is_cuda());
    TORCH_CHECK(csr_off.scalar_type() == torch::kInt32 && csr_idx.scalar_type() == torch::kInt32);
    TORCH_CHECK(csr_off.dim() == 2 && csr_idx.dim() == 2);
    TORCH_CHECK(csr_idx.size(1) == n);
    const int off_stride = static_cast<int>(csr_off.size(1));
    int cache_y = 0;
    const int cap = dyn_smem_cap();
    TORCH_CHECK(strip_hist_bytes(V, C) <= static_cast<size_t>(cap), "χ² k=0 V×C hist exceeds SMEM");
    const size_t smem = k0_tiled_smem(V, C, n, cap, &cache_y);
    TORCH_CHECK(static_cast<int>(smem) <= cap, "χ² k=0 strip smem exceeds opt-in");
    launch_k0_tiled_v(V, data.data_ptr<Code>(), cards.data_ptr<int>(), x.data_ptr<int>(),
                      y.data_ptr<int>(), stat.data_ptr<double>(), df.data_ptr<int>(),
                      csr_off.data_ptr<int>(), csr_idx.data_ptr<int>(), off_stride, C, n, cache_y,
                      B, smem, stream);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void chisq_k1(torch::Tensor data, torch::Tensor cards, torch::Tensor x, torch::Tensor y,
              torch::Tensor z, torch::Tensor stat, torch::Tensor df, int64_t z_cap,
              int64_t max_card, torch::Tensor csr_off, torch::Tensor csr_idx, int64_t tile_v) {
    check_codes(data);
    TORCH_CHECK(cards.is_cuda() && cards.scalar_type() == torch::kInt32, "cards int32 CUDA");
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && z.is_cuda() && stat.is_cuda() && df.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kInt32 && y.scalar_type() == torch::kInt32);
    TORCH_CHECK(z.scalar_type() == torch::kInt32);
    TORCH_CHECK(stat.scalar_type() == torch::kFloat64 && df.scalar_type() == torch::kInt32);
    (void)z_cap;
    const int64_t B = x.numel();
    TORCH_CHECK(y.numel() == B && z.numel() == B && stat.numel() == B && df.numel() == B);
    const int n = static_cast<int>(data.size(1));
    if (B == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int requested = static_cast<int>(tile_v);
    TORCH_CHECK(max_card >= 1 && max_card <= kCFull, "max_card must be in 1..1024");
    const int C = static_cast<int>(max_card);
    const int V = resolve_tile_v(C, requested);
    TORCH_CHECK(csr_off.is_cuda() && csr_idx.is_cuda());
    TORCH_CHECK(csr_off.scalar_type() == torch::kInt32 && csr_idx.scalar_type() == torch::kInt32);
    const int off_stride = static_cast<int>(csr_off.size(1));
    int cache_xy = 0;
    int local_csr = 0;
    const int dyn_cap = dyn_smem_cap();
    TORCH_CHECK(strip_hist_bytes(V, C) <= static_cast<size_t>(dyn_cap),
                "χ² k=1 V×C hist exceeds SMEM");
    const size_t smem = k1_tiled_smem(V, C, n, dyn_cap, &cache_xy, &local_csr);
    launch_k1_tiled_v(V, data.data_ptr<Code>(), cards.data_ptr<int>(), x.data_ptr<int>(),
                      y.data_ptr<int>(), z.data_ptr<int>(), stat.data_ptr<double>(),
                      df.data_ptr<int>(), csr_off.data_ptr<int>(), csr_idx.data_ptr<int>(),
                      off_stride, C, n, cache_xy, local_csr, B, smem, stream);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void chisq_k1_gmem(torch::Tensor data, torch::Tensor cards, torch::Tensor x, torch::Tensor y,
                   torch::Tensor z, torch::Tensor stat, torch::Tensor df, int64_t max_card,
                   torch::Tensor csr_off, torch::Tensor csr_idx, int64_t tile_v,
                   torch::Tensor ws_loc, int64_t grid) {
    check_codes(data);
    TORCH_CHECK(cards.is_cuda() && cards.scalar_type() == torch::kInt32, "cards int32 CUDA");
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && z.is_cuda() && stat.is_cuda() && df.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kInt32 && y.scalar_type() == torch::kInt32);
    TORCH_CHECK(z.scalar_type() == torch::kInt32);
    TORCH_CHECK(stat.scalar_type() == torch::kFloat64 && df.scalar_type() == torch::kInt32);
    const int64_t B = x.numel();
    TORCH_CHECK(y.numel() == B && z.numel() == B && stat.numel() == B && df.numel() == B);
    const int n = static_cast<int>(data.size(1));
    if (B == 0) {
        return;
    }
    TORCH_CHECK(grid >= 1, "grid must be positive");
    const int64_t g = grid < B ? grid : B;
    TORCH_CHECK(ws_loc.is_cuda() && ws_loc.scalar_type() == torch::kInt32);
    TORCH_CHECK(ws_loc.is_contiguous() && ws_loc.numel() >= g * n, "workspace too small");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int requested = static_cast<int>(tile_v);
    TORCH_CHECK(max_card >= 1 && max_card <= kCFull, "max_card must be in 1..1024");
    const int C = static_cast<int>(max_card);
    const int V = resolve_tile_v(C, requested);
    TORCH_CHECK(csr_off.is_cuda() && csr_idx.is_cuda());
    TORCH_CHECK(csr_off.scalar_type() == torch::kInt32 && csr_idx.scalar_type() == torch::kInt32);
    const int off_stride = static_cast<int>(csr_off.size(1));
    int cache_xy = 0;
    const int dyn_cap = dyn_smem_cap();
    TORCH_CHECK(strip_hist_bytes(V, C) <= static_cast<size_t>(dyn_cap),
                "χ² k=1 V×C hist exceeds SMEM");
    const size_t smem = k1_gmem_smem(V, C, n, dyn_cap, &cache_xy);
    launch_k1_gmem_v(V, data.data_ptr<Code>(), cards.data_ptr<int>(), x.data_ptr<int>(),
                     y.data_ptr<int>(), z.data_ptr<int>(), stat.data_ptr<double>(),
                     df.data_ptr<int>(), csr_off.data_ptr<int>(), csr_idx.data_ptr<int>(),
                     off_stride, C, n, cache_xy, B, ws_loc.data_ptr<int>(), static_cast<int>(g),
                     smem, stream);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void chisq_ks(torch::Tensor data, torch::Tensor cards, torch::Tensor x, torch::Tensor y,
              torch::Tensor S, torch::Tensor stat, torch::Tensor df, int64_t k, int64_t z_cap,
              int64_t max_card, int64_t tile_v) {
    check_codes(data);
    TORCH_CHECK(cards.is_cuda() && cards.scalar_type() == torch::kInt32, "cards int32 CUDA");
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && S.is_cuda() && stat.is_cuda() && df.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kInt32 && y.scalar_type() == torch::kInt32);
    TORCH_CHECK(S.scalar_type() == torch::kInt32 && S.dim() == 2);
    TORCH_CHECK(stat.scalar_type() == torch::kFloat64 && df.scalar_type() == torch::kInt32);
    const int64_t B = x.numel();
    TORCH_CHECK(y.numel() == B && S.size(0) == B && stat.numel() == B && df.numel() == B);
    TORCH_CHECK(k >= 2 && k <= S.size(1), "k must be in 2..S.size(1)");
    const int n = static_cast<int>(data.size(1));
    if (B == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int s_stride = static_cast<int>(S.size(1));
    const int requested = static_cast<int>(tile_v);
    TORCH_CHECK(max_card >= 1 && max_card <= kCFull, "max_card must be in 1..1024");
    const int C = static_cast<int>(max_card);
    const int V = resolve_tile_v(C, requested);
    int cache_xy = 0;
    const int dyn_cap = dyn_smem_cap();
    TORCH_CHECK(strip_hist_bytes(V, C) <= static_cast<size_t>(dyn_cap),
                "χ² packed V×C hist exceeds SMEM");
    const size_t smem = k_pack_tiled_smem(V, C, n, dyn_cap, &cache_xy);
    TORCH_CHECK(smem > 0, "χ² packed strip smem exceeds opt-in");
    (void)z_cap;
    launch_pack_tiled_v(V, data.data_ptr<Code>(), cards.data_ptr<int>(), x.data_ptr<int>(),
                        y.data_ptr<int>(), S.data_ptr<int>(), stat.data_ptr<double>(),
                        df.data_ptr<int>(), C, n, static_cast<int>(k), s_stride, cache_xy, B, smem,
                        stream);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

void chisq_ks_gmem(torch::Tensor data, torch::Tensor cards, torch::Tensor x, torch::Tensor y,
                   torch::Tensor S, torch::Tensor stat, torch::Tensor df, int64_t k,
                   int64_t max_card, int64_t tile_v, torch::Tensor ws_z, torch::Tensor ws_x,
                   int64_t grid) {
    check_codes(data);
    TORCH_CHECK(cards.is_cuda() && cards.scalar_type() == torch::kInt32, "cards int32 CUDA");
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && S.is_cuda() && stat.is_cuda() && df.is_cuda());
    TORCH_CHECK(x.scalar_type() == torch::kInt32 && y.scalar_type() == torch::kInt32);
    TORCH_CHECK(S.scalar_type() == torch::kInt32 && S.dim() == 2);
    TORCH_CHECK(stat.scalar_type() == torch::kFloat64 && df.scalar_type() == torch::kInt32);
    const int64_t B = x.numel();
    TORCH_CHECK(y.numel() == B && S.size(0) == B && stat.numel() == B && df.numel() == B);
    TORCH_CHECK(k >= 2 && k <= S.size(1), "k must be in 2..S.size(1)");
    const int n = static_cast<int>(data.size(1));
    if (B == 0) {
        return;
    }
    TORCH_CHECK(grid >= 1, "grid must be positive");
    const int64_t g = grid < B ? grid : B;
    TORCH_CHECK(ws_z.is_cuda() && ws_x.is_cuda(), "workspaces must be CUDA");
    TORCH_CHECK(ws_z.scalar_type() == torch::kInt32 && ws_x.scalar_type() == torch::kInt32);
    TORCH_CHECK(ws_z.is_contiguous() && ws_x.is_contiguous(), "workspaces must be contiguous");
    TORCH_CHECK(ws_z.numel() >= g * n && ws_x.numel() >= g * n, "workspace too small");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int s_stride = static_cast<int>(S.size(1));
    const int requested = static_cast<int>(tile_v);
    TORCH_CHECK(max_card >= 1 && max_card <= kCFull, "max_card must be in 1..1024");
    const int C = static_cast<int>(max_card);
    const int V = resolve_tile_v(C, requested);
    int cache_xy = 0;
    const int dyn_cap = dyn_smem_cap();
    TORCH_CHECK(strip_hist_bytes(V, C) <= static_cast<size_t>(dyn_cap),
                "χ² packed V×C hist exceeds SMEM");
    const size_t smem = k_pack_gmem_smem(V, C, n, dyn_cap, &cache_xy);
    launch_pack_gmem_v(V, data.data_ptr<Code>(), cards.data_ptr<int>(), x.data_ptr<int>(),
                       y.data_ptr<int>(), S.data_ptr<int>(), stat.data_ptr<double>(),
                       df.data_ptr<int>(), C, n, static_cast<int>(k), s_stride, cache_xy, B,
                       ws_z.data_ptr<int>(), ws_x.data_ptr<int>(), static_cast<int>(g), smem,
                       stream);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

int chisq_c_full() { return kCFull; }

int chisq_tile_v_for_card(int64_t C) { return tile_v_for_card(static_cast<int>(C)); }

int chisq_optin_smem() { return optin_dyn_smem(); }

void chisq_preload() {
    cuda_preload_context();
    touch_cuda_kernel(chisq_k0_tiled_kernel<4>);
    touch_cuda_kernel(chisq_k0_tiled_kernel<8>);
    touch_cuda_kernel(chisq_k0_tiled_kernel<16>);
    touch_cuda_kernel(chisq_k0_tiled_kernel<32>);
    touch_cuda_kernel(chisq_k1_tiled_kernel<4>);
    touch_cuda_kernel(chisq_k1_tiled_kernel<8>);
    touch_cuda_kernel(chisq_k1_tiled_kernel<16>);
    touch_cuda_kernel(chisq_k1_tiled_kernel<32>);
    touch_cuda_kernel(chisq_k1_gmem_kernel<4>);
    touch_cuda_kernel(chisq_k1_gmem_kernel<8>);
    touch_cuda_kernel(chisq_k1_gmem_kernel<16>);
    touch_cuda_kernel(chisq_k1_gmem_kernel<32>);
    touch_cuda_kernel(chisq_k_pack_tiled_kernel<4>);
    touch_cuda_kernel(chisq_k_pack_tiled_kernel<8>);
    touch_cuda_kernel(chisq_k_pack_tiled_kernel<16>);
    touch_cuda_kernel(chisq_k_pack_tiled_kernel<32>);
    touch_cuda_kernel(chisq_k_pack_gmem_kernel<4>);
    touch_cuda_kernel(chisq_k_pack_gmem_kernel<8>);
    touch_cuda_kernel(chisq_k_pack_gmem_kernel<16>);
    touch_cuda_kernel(chisq_k_pack_gmem_kernel<32>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k0", &chisq_k0, pybind11::arg("data"), pybind11::arg("cards"), pybind11::arg("x"),
          pybind11::arg("y"), pybind11::arg("stat"), pybind11::arg("df"), pybind11::arg("max_card"),
          pybind11::arg("csr_off"), pybind11::arg("csr_idx"), pybind11::arg("tile_v") = 0);
    m.def("k1", &chisq_k1, pybind11::arg("data"), pybind11::arg("cards"), pybind11::arg("x"),
          pybind11::arg("y"), pybind11::arg("z"), pybind11::arg("stat"), pybind11::arg("df"),
          pybind11::arg("z_cap"), pybind11::arg("max_card"), pybind11::arg("csr_off"),
          pybind11::arg("csr_idx"), pybind11::arg("tile_v") = 0);
    m.def("ks", &chisq_ks, pybind11::arg("data"), pybind11::arg("cards"), pybind11::arg("x"),
          pybind11::arg("y"), pybind11::arg("S"), pybind11::arg("stat"), pybind11::arg("df"),
          pybind11::arg("k"), pybind11::arg("z_cap"), pybind11::arg("max_card"),
          pybind11::arg("tile_v") = 0);
    m.def("k1_gmem", &chisq_k1_gmem, pybind11::arg("data"), pybind11::arg("cards"),
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"), pybind11::arg("stat"),
          pybind11::arg("df"), pybind11::arg("max_card"), pybind11::arg("csr_off"),
          pybind11::arg("csr_idx"), pybind11::arg("tile_v"), pybind11::arg("ws_loc"),
          pybind11::arg("grid"));
    m.def("ks_gmem", &chisq_ks_gmem, pybind11::arg("data"), pybind11::arg("cards"),
          pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("S"), pybind11::arg("stat"),
          pybind11::arg("df"), pybind11::arg("k"), pybind11::arg("max_card"),
          pybind11::arg("tile_v"), pybind11::arg("ws_z"), pybind11::arg("ws_x"),
          pybind11::arg("grid"));
    m.def("c_full", &chisq_c_full);
    m.def("tile_v_for_card", &chisq_tile_v_for_card);
    m.def("static_smem_reserve", []() { return kStaticReserve; });
    m.def("optin_smem", &chisq_optin_smem);
    m.def("preload", &chisq_preload);
}

} // namespace
