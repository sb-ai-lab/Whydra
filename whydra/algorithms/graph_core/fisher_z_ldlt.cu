// Batched Fisher-Z via unpivoted LDLᵀ of G = C[Z,Z].
//
// Template policy:
//   T              — double (float if WHYDRA_ENABLE_FLOAT32=1)
//   TPT            — warp threads that cooperate on one batch element
//                    (must divide 32; groups are warp-aligned)
//   K_MAX          — compile-time cap for the generic kernel; runtime k <= K_MAX
//
// Specializations k = 0,1,2,3 are fully unrolled (registers, TPT=1 by default).
// Generic k uses K_MAX + runtime k, shared memory, TPT from tpt_for_k(k).
// Large k packs fewer tests per CTA so the k×k factor fits opt-in smem.
//
// Device inputs: C (any 2D strides, element units) + index arrays x, y, Z.
// Each test gathers C[i,j] = C[i * s0 + j * s1] into registers / smem.
// L [ux uy] = [ax ay], scale one RHS by (D+eps)^{-1},
// r from α − uᵀ (D+eps)^{-1} v.
// det(D + u vᵀ) = det(D) (1 + vᵀ D^{-1} u)  [matrix determinant lemma]. The
// optional eps scales every D^{-1} by (D+eps)^{-1} — the classical modified-
// Cholesky diagonal modification (Gill-Murray-Wright 1981; Schnabel-Eskow
// 1990), guarding zero pivots of singular G = C[Z,Z]. Default eps = 0 (exact).
//
// Entry points (exported as ldlt_batch / ldlt_factor / ldlt_apply / k_max /
// tpt_for_k / preload by the PYBIND11_MODULE at the bottom):
//   fisher_z_ldlt_batch(C, x, y, Z, out, eps)     one partial r per test
//   fisher_z_ldlt_factor(C, Z, L, D, eps)         factor U unique Z blocks
//   fisher_z_ldlt_apply(C, x, y, Z, inv, L, D, out, eps)
//                                                 apply a cached factor; inv
//                                                 maps each test to its Z row
//   fisher_z_ldlt_k_max / fisher_z_ldlt_tpt      device smem cap / warp layout
//
// Memory / layout model: C is any 2-D strides s0, s1 in element units; Z is
// int32 [B, k] (conditioning sets, one row per test); x/y are int32 [B]; out is
// the same floating dtype as C. The factor buffers are [U, k, k] / [U, k] for
// U unique conditioning sets. Nothing here touches the graph adjacency.
//
// Invariants: the returned value is the Fisher-Z partial correlation (not a
// p-value); NaN/inf propagates and is rejected by the caller. k <= kMaxCond.
// The math matches graph_batch.cu's r_cond/r_cond_generic so the pair-major
// and flat CI kernels produce bit-compatible results for the same inputs.
//
// Python counterparts: cuda_fisher_z.py (CudaFisherZ.batch_r / batch) and
// graph_core/cuda_graph_batch.py (CI batches that call this extension).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_check.cuh"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_float.cuh"
#include "cuda_touch.cuh"

namespace {

constexpr int kMaxCond = 96;
constexpr int kBlock = 256;

template <typename T> __device__ __forceinline__ T dsqrt(T x);

template <> __device__ __forceinline__ float dsqrt<float>(float x) { return sqrtf(x); }
template <> __device__ __forceinline__ double dsqrt<double>(double x) { return sqrt(x); }

template <typename T> __device__ __forceinline__ T dfabs(T x);

template <> __device__ __forceinline__ float dfabs<float>(float x) { return fabsf(x); }
template <> __device__ __forceinline__ double dfabs<double>(double x) { return fabs(x); }

template <typename T>
__device__ __forceinline__ T c_at(const T* C, int i, int j, long long s0, long long s1) {
    return __ldg(C + static_cast<long long>(i) * s0 + static_cast<long long>(j) * s1);
}

template <typename T> __device__ __forceinline__ T r_from_dets(T qxy, T qxx, T qyy) {
    return qxy / dsqrt(dfabs(qxx * qyy));
}

// Scale one L-solve by (D+eps)^{-1}. t==0 only.
template <typename T>
__device__ __forceinline__ T r_from_ux_uy(T axx, T ayy, T axy, const T* ux, const T* uy, const T* D,
                                          int k, T eps) {
    T qxy = axy;
    T qxx = axx;
    T qyy = ayy;
    for (int i = 0; i < k; ++i) {
        const T inv = T(1) / (D[i] + eps);
        qxy -= ux[i] * uy[i] * inv;
        qxx -= ux[i] * ux[i] * inv;
        qyy -= uy[i] * uy[i] * inv;
    }
    return r_from_dets(qxy, qxx, qyy);
}

template <typename T> __device__ __forceinline__ void solve_unit_lower(const T* L, T* u, int k) {
    for (int i = 1; i < k; ++i) {
        T s = u[i];
        for (int j = 0; j < i; ++j) {
            s -= L[i * k + j] * u[j];
        }
        u[i] = s;
    }
}

// ---- k = 0, 1, 2, 3  (compile-time k, TPT threads per test) ----

template <typename T, int TPT>
__global__ void fisher_z_k0(const T* __restrict__ C, long long s0, long long s1,
                            const int* __restrict__ x, const int* __restrict__ y,
                            T* __restrict__ out, int B) {
    static_assert(TPT >= 1 && TPT <= 32 && (32 % TPT) == 0, "TPT must divide 32");
    const int tests_per_cta = kBlock / TPT;
    const int local = static_cast<int>(threadIdx.x) / TPT;
    const int t = static_cast<int>(threadIdx.x) % TPT;
    const int b = static_cast<int>(blockIdx.x) * tests_per_cta + local;
    if (t != 0 || b >= B) {
        return;
    }
    const int xi = x[b];
    const int yi = y[b];
    const T cxy = c_at(C, xi, yi, s0, s1);
    const T cxx = c_at(C, xi, xi, s0, s1);
    const T cyy = c_at(C, yi, yi, s0, s1);
    out[b] = cxy / dsqrt(dfabs(cxx * cyy));
}

template <typename T, int TPT>
__global__ void fisher_z_k1(const T* __restrict__ C, long long s0, long long s1,
                            const int* __restrict__ x, const int* __restrict__ y,
                            const int* __restrict__ Z, T* __restrict__ out, int B, T eps) {
    static_assert(TPT >= 1 && TPT <= 32 && (32 % TPT) == 0, "TPT must divide 32");
    const int tests_per_cta = kBlock / TPT;
    const int local = static_cast<int>(threadIdx.x) / TPT;
    const int t = static_cast<int>(threadIdx.x) % TPT;
    const int b = static_cast<int>(blockIdx.x) * tests_per_cta + local;
    if (t != 0 || b >= B) {
        return;
    }
    const int xi = x[b];
    const int yi = y[b];
    const int s = Z[b];
    const T cxx = c_at(C, xi, xi, s0, s1);
    const T cyy = c_at(C, yi, yi, s0, s1);
    const T css = c_at(C, s, s, s0, s1);
    const T cxy = c_at(C, xi, yi, s0, s1);
    const T cxs = c_at(C, xi, s, s0, s1);
    const T cys = c_at(C, yi, s, s0, s1);
    const T d = css + eps;
    const T inv = T(1) / d;
    out[b] = r_from_dets(cxy - cxs * cys * inv, cxx - cxs * cxs * inv, cyy - cys * cys * inv);
}

template <typename T, int TPT>
__global__ void fisher_z_k2(const T* __restrict__ C, long long s0, long long s1,
                            const int* __restrict__ x, const int* __restrict__ y,
                            const int* __restrict__ Z, T* __restrict__ out, int B, T eps) {
    static_assert(TPT >= 1 && TPT <= 32 && (32 % TPT) == 0, "TPT must divide 32");
    const int tests_per_cta = kBlock / TPT;
    const int local = static_cast<int>(threadIdx.x) / TPT;
    const int t = static_cast<int>(threadIdx.x) % TPT;
    const int b = static_cast<int>(blockIdx.x) * tests_per_cta + local;
    if (t != 0 || b >= B) {
        return;
    }
    const int xi = x[b];
    const int yi = y[b];
    const int z0 = Z[b * 2];
    const int z1 = Z[b * 2 + 1];
    const T g00 = c_at(C, z0, z0, s0, s1);
    const T g01 = c_at(C, z0, z1, s0, s1);
    const T g11 = c_at(C, z1, z1, s0, s1);
    const T ax0 = c_at(C, z0, xi, s0, s1);
    const T ax1 = c_at(C, z1, xi, s0, s1);
    const T ay0 = c_at(C, z0, yi, s0, s1);
    const T ay1 = c_at(C, z1, yi, s0, s1);
    const T axx = c_at(C, xi, xi, s0, s1);
    const T ayy = c_at(C, yi, yi, s0, s1);
    const T axy = c_at(C, xi, yi, s0, s1);
    const T D0 = g00;
    const T L10 = (D0 != T(0)) ? g01 / D0 : T(0);
    const T D1 = g11 - L10 * L10 * D0;
    const T ux0 = ax0;
    const T ux1 = ax1 - L10 * ux0;
    const T uy0 = ay0;
    const T uy1 = ay1 - L10 * uy0;
    const T dp0 = D0 + eps;
    const T dp1 = D1 + eps;
    const T inv0 = T(1) / dp0;
    const T inv1 = T(1) / dp1;
    const T qxy = axy - ux0 * uy0 * inv0 - ux1 * uy1 * inv1;
    const T qxx = axx - ux0 * ux0 * inv0 - ux1 * ux1 * inv1;
    const T qyy = ayy - uy0 * uy0 * inv0 - uy1 * uy1 * inv1;
    out[b] = r_from_dets(qxy, qxx, qyy);
}

template <typename T, int TPT>
__global__ void fisher_z_k3(const T* __restrict__ C, long long s0, long long s1,
                            const int* __restrict__ x, const int* __restrict__ y,
                            const int* __restrict__ Z, T* __restrict__ out, int B, T eps) {
    static_assert(TPT >= 1 && TPT <= 32 && (32 % TPT) == 0, "TPT must divide 32");
    const int tests_per_cta = kBlock / TPT;
    const int local = static_cast<int>(threadIdx.x) / TPT;
    const int t = static_cast<int>(threadIdx.x) % TPT;
    const int b = static_cast<int>(blockIdx.x) * tests_per_cta + local;
    if (t != 0 || b >= B) {
        return;
    }
    const int xi = x[b];
    const int yi = y[b];
    const int z0 = Z[b * 3];
    const int z1 = Z[b * 3 + 1];
    const int z2 = Z[b * 3 + 2];
    const T g00 = c_at(C, z0, z0, s0, s1);
    const T g01 = c_at(C, z0, z1, s0, s1);
    const T g02 = c_at(C, z0, z2, s0, s1);
    const T g11 = c_at(C, z1, z1, s0, s1);
    const T g12 = c_at(C, z1, z2, s0, s1);
    const T g22 = c_at(C, z2, z2, s0, s1);
    const T ax0 = c_at(C, z0, xi, s0, s1);
    const T ax1 = c_at(C, z1, xi, s0, s1);
    const T ax2 = c_at(C, z2, xi, s0, s1);
    const T ay0 = c_at(C, z0, yi, s0, s1);
    const T ay1 = c_at(C, z1, yi, s0, s1);
    const T ay2 = c_at(C, z2, yi, s0, s1);
    const T axx = c_at(C, xi, xi, s0, s1);
    const T ayy = c_at(C, yi, yi, s0, s1);
    const T axy = c_at(C, xi, yi, s0, s1);
    const T D0 = g00;
    const T L10 = (D0 != T(0)) ? g01 / D0 : T(0);
    const T L20 = (D0 != T(0)) ? g02 / D0 : T(0);
    const T D1 = g11 - L10 * L10 * D0;
    const T L21 = (D1 != T(0)) ? (g12 - L20 * L10 * D0) / D1 : T(0);
    const T D2 = g22 - L20 * L20 * D0 - L21 * L21 * D1;
    const T ux0 = ax0;
    const T ux1 = ax1 - L10 * ux0;
    const T ux2 = ax2 - L20 * ux0 - L21 * ux1;
    const T uy0 = ay0;
    const T uy1 = ay1 - L10 * uy0;
    const T uy2 = ay2 - L20 * uy0 - L21 * uy1;
    const T dp0 = D0 + eps;
    const T dp1 = D1 + eps;
    const T dp2 = D2 + eps;
    const T inv0 = T(1) / dp0;
    const T inv1 = T(1) / dp1;
    const T inv2 = T(1) / dp2;
    const T qxy = axy - ux0 * uy0 * inv0 - ux1 * uy1 * inv1 - ux2 * uy2 * inv2;
    const T qxx = axx - ux0 * ux0 * inv0 - ux1 * ux1 * inv1 - ux2 * ux2 * inv2;
    const T qyy = ayy - uy0 * uy0 * inv0 - uy1 * uy1 * inv1 - uy2 * uy2 * inv2;
    out[b] = r_from_dets(qxy, qxx, qyy);
}

// ---- generic: K_MAX compile-time, k runtime, TPT cooperative loads ----

template <typename T, int K_MAX, int TPT>
__global__ void fisher_z_ldlt_generic(const T* __restrict__ C, long long s0, long long s1,
                                      const int* __restrict__ x, const int* __restrict__ y,
                                      const int* __restrict__ Z, T* __restrict__ out, int B, int k,
                                      T eps) {
    static_assert(K_MAX >= 4 && K_MAX <= 96, "K_MAX in [4, 96]");
    static_assert(TPT >= 1 && TPT <= 32 && (32 % TPT) == 0, "TPT must divide 32");
    const int tests_per_cta = static_cast<int>(blockDim.x) / TPT;
    const int local = static_cast<int>(threadIdx.x) / TPT;
    const int t = static_cast<int>(threadIdx.x) % TPT;
    const int b = static_cast<int>(blockIdx.x) * tests_per_cta + local;
    const bool live = b < B;

    extern __shared__ unsigned char raw[];
    T* smem_T = reinterpret_cast<T*>(raw);
    const int t_stride = k * k + 3 * k;
    T* L = smem_T + static_cast<int>(local) * t_stride;
    T* D = L + k * k;
    T* ux = D + k;
    T* uy = ux + k;
    int* smem_i = reinterpret_cast<int*>(smem_T + tests_per_cta * t_stride);
    int* z = smem_i + static_cast<int>(local) * k;

    if (live) {
        for (int i = t; i < k; i += TPT) {
            z[i] = Z[b * k + i];
        }
    }
    __syncwarp();

    if (live) {
        const int xi = x[b];
        const int yi = y[b];
        for (int idx = t; idx < k * k; idx += TPT) {
            const int i = idx / k;
            const int j = idx - i * k;
            L[idx] = c_at(C, z[i], z[j], s0, s1);
        }
        for (int i = t; i < k; i += TPT) {
            ux[i] = c_at(C, z[i], xi, s0, s1);
            uy[i] = c_at(C, z[i], yi, s0, s1);
        }
    }
    __syncwarp();

    // Idle groups (b >= B) still take the k-loop and warp syncs.
    for (int j = 0; j < k; ++j) {
        if (live && t == 0) {
            T acc = L[j * k + j];
            for (int p = 0; p < j; ++p) {
                const T lp = L[j * k + p];
                acc -= lp * lp * D[p];
            }
            D[j] = acc;
            L[j * k + j] = T(1);
        }
        __syncwarp();
        if (live) {
            const T acc = D[j];
            for (int i = j + 1 + t; i < k; i += TPT) {
                T s = L[i * k + j];
                for (int p = 0; p < j; ++p) {
                    s -= L[i * k + p] * L[j * k + p] * D[p];
                }
                L[i * k + j] = (acc != T(0)) ? s / acc : T(0);
            }
        }
        __syncwarp();
    }

    if (t == 0 && live) {
        solve_unit_lower(L, ux, k);
        solve_unit_lower(L, uy, k);
        const int xi = x[b];
        const int yi = y[b];
        const T axx = c_at(C, xi, xi, s0, s1);
        const T ayy = c_at(C, yi, yi, s0, s1);
        const T axy = c_at(C, xi, yi, s0, s1);
        out[b] = r_from_ux_uy(axx, ayy, axy, ux, uy, D, k, eps);
    }
}

inline int tpt_for_k(int k) {
    // Must divide 32 (warp-aligned groups). k>16 uses a full warp per test.
    if (k <= 3) {
        return 1;
    }
    if (k <= 8) {
        return 8;
    }
    if (k <= 16) {
        return 16;
    }
    return 32;
}

inline int grid_for_B(int B, int tpt) {
    const int tests_per_cta = kBlock / tpt;
    return (B + tests_per_cta - 1) / tests_per_cta;
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

inline int tests_per_cta_fit(int k, int tpt, size_t t_bytes, size_t t_stride) {
    const size_t per = t_stride * t_bytes + static_cast<size_t>(k) * sizeof(int);
    const int cap = optin_dyn_smem();
    int by_smem = per == 0 ? (kBlock / tpt) : static_cast<int>(cap / per);
    if (by_smem < 1) {
        by_smem = 1;
    }
    const int by_block = kBlock / tpt;
    return by_smem < by_block ? by_smem : by_block;
}

inline size_t generic_smem_bytes(int k, int tests_per_cta, size_t t_bytes) {
    const size_t t_stride = static_cast<size_t>(k * k + 3 * k);
    return static_cast<size_t>(tests_per_cta) *
           (t_stride * t_bytes + static_cast<size_t>(k) * sizeof(int));
}

inline size_t factor_smem_bytes(int k, int tests_per_cta, size_t t_bytes) {
    const size_t t_stride = static_cast<size_t>(k * k + k);
    return static_cast<size_t>(tests_per_cta) *
           (t_stride * t_bytes + static_cast<size_t>(k) * sizeof(int));
}

template <typename Fn> void set_dyn_smem(Fn* fn, size_t smem) {
    const int cap = optin_dyn_smem();
    TORCH_CHECK(static_cast<int>(smem) <= cap, "Fisher-Z dynamic smem ", smem,
                " exceeds device opt-in ", cap);
    if (smem > 48 * 1024) {
        TORCH_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)) == cudaSuccess,
                    "cudaFuncSetAttribute max dynamic smem");
    }
}

template <typename T, int TPT>
void launch_generic(cudaStream_t stream, const T* C, long long s0, long long s1, const int* x,
                    const int* y, const int* Z, T* out, int B, int k, T eps) {
    const size_t t_stride = static_cast<size_t>(k * k + 3 * k);
    const int tpc = tests_per_cta_fit(k, TPT, sizeof(T), t_stride);
    const int block = TPT * tpc;
    const int blocks = (B + tpc - 1) / tpc;
    const size_t smem = generic_smem_bytes(k, tpc, sizeof(T));
    auto* fn = fisher_z_ldlt_generic<T, kMaxCond, TPT>;
    set_dyn_smem(fn, smem);
    fn<<<blocks, block, smem, stream>>>(C, s0, s1, x, y, Z, out, B, k, eps);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void dispatch_ldlt(cudaStream_t stream, const T* C, long long s0, long long s1, const int* x,
                   const int* y, const int* Z, T* out, int B, int k, T eps) {
    const int tpt = tpt_for_k(k);
    const int blocks = grid_for_B(B, tpt);
    switch (k) {
    case 0:
        fisher_z_k0<T, 1><<<blocks, kBlock, 0, stream>>>(C, s0, s1, x, y, out, B);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    case 1:
        fisher_z_k1<T, 1><<<blocks, kBlock, 0, stream>>>(C, s0, s1, x, y, Z, out, B, eps);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    case 2:
        fisher_z_k2<T, 1><<<blocks, kBlock, 0, stream>>>(C, s0, s1, x, y, Z, out, B, eps);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    case 3:
        fisher_z_k3<T, 1><<<blocks, kBlock, 0, stream>>>(C, s0, s1, x, y, Z, out, B, eps);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    default:
        break;
    }
    switch (tpt) {
    case 8:
        launch_generic<T, 8>(stream, C, s0, s1, x, y, Z, out, B, k, eps);
        break;
    case 16:
        launch_generic<T, 16>(stream, C, s0, s1, x, y, Z, out, B, k, eps);
        break;
    case 32:
        launch_generic<T, 32>(stream, C, s0, s1, x, y, Z, out, B, k, eps);
        break;
    default:
        launch_generic<T, 8>(stream, C, s0, s1, x, y, Z, out, B, k, eps);
        break;
    }
}

// ---- unique-S factor + apply (one LDLᵀ of C_SS per conditioner) ----

template <typename T>
__global__ void ldlt_factor_k2(const T* __restrict__ C, long long s0, long long s1,
                               const int* __restrict__ Z, T* __restrict__ L, T* __restrict__ D,
                               int U) {
    const int u = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    if (u >= U) {
        return;
    }
    const int z0 = Z[u * 2];
    const int z1 = Z[u * 2 + 1];
    const T g00 = c_at(C, z0, z0, s0, s1);
    const T g01 = c_at(C, z0, z1, s0, s1);
    const T g11 = c_at(C, z1, z1, s0, s1);
    const T D0 = g00;
    const T L10 = (D0 != T(0)) ? g01 / D0 : T(0);
    const T D1 = g11 - L10 * L10 * D0;
    L[u * 4 + 0] = T(1);
    L[u * 4 + 1] = T(0);
    L[u * 4 + 2] = L10;
    L[u * 4 + 3] = T(1);
    D[u * 2 + 0] = D0;
    D[u * 2 + 1] = D1;
}

template <typename T>
__global__ void
ldlt_apply_k2(const T* __restrict__ C, long long s0, long long s1, const int* __restrict__ x,
              const int* __restrict__ y, const int* __restrict__ Z, const int* __restrict__ inv,
              const T* __restrict__ L, const T* __restrict__ D, T* __restrict__ out, int B, T eps) {
    const int b = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    if (b >= B) {
        return;
    }
    const int u = inv[b];
    const int xi = x[b];
    const int yi = y[b];
    const int z0 = Z[u * 2];
    const int z1 = Z[u * 2 + 1];
    const T L10 = L[u * 4 + 2];
    const T D0 = D[u * 2];
    const T D1 = D[u * 2 + 1];
    const T ax0 = c_at(C, z0, xi, s0, s1);
    const T ax1 = c_at(C, z1, xi, s0, s1);
    const T ay0 = c_at(C, z0, yi, s0, s1);
    const T ay1 = c_at(C, z1, yi, s0, s1);
    const T axx = c_at(C, xi, xi, s0, s1);
    const T ayy = c_at(C, yi, yi, s0, s1);
    const T axy = c_at(C, xi, yi, s0, s1);
    const T ux0 = ax0;
    const T ux1 = ax1 - L10 * ux0;
    const T uy0 = ay0;
    const T uy1 = ay1 - L10 * uy0;
    const T inv0 = T(1) / (D0 + eps);
    const T inv1 = T(1) / (D1 + eps);
    const T qxy = axy - ux0 * uy0 * inv0 - ux1 * uy1 * inv1;
    const T qxx = axx - ux0 * ux0 * inv0 - ux1 * ux1 * inv1;
    const T qyy = ayy - uy0 * uy0 * inv0 - uy1 * uy1 * inv1;
    out[b] = r_from_dets(qxy, qxx, qyy);
}

template <typename T>
__global__ void ldlt_factor_k3(const T* __restrict__ C, long long s0, long long s1,
                               const int* __restrict__ Z, T* __restrict__ L, T* __restrict__ D,
                               int U) {
    const int u = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    if (u >= U) {
        return;
    }
    const int z0 = Z[u * 3];
    const int z1 = Z[u * 3 + 1];
    const int z2 = Z[u * 3 + 2];
    const T g00 = c_at(C, z0, z0, s0, s1);
    const T g01 = c_at(C, z0, z1, s0, s1);
    const T g02 = c_at(C, z0, z2, s0, s1);
    const T g11 = c_at(C, z1, z1, s0, s1);
    const T g12 = c_at(C, z1, z2, s0, s1);
    const T g22 = c_at(C, z2, z2, s0, s1);
    const T D0 = g00;
    const T L10 = (D0 != T(0)) ? g01 / D0 : T(0);
    const T L20 = (D0 != T(0)) ? g02 / D0 : T(0);
    const T D1 = g11 - L10 * L10 * D0;
    const T L21 = (D1 != T(0)) ? (g12 - L20 * L10 * D0) / D1 : T(0);
    const T D2 = g22 - L20 * L20 * D0 - L21 * L21 * D1;
    const int o = u * 9;
    L[o + 0] = T(1);
    L[o + 1] = T(0);
    L[o + 2] = T(0);
    L[o + 3] = L10;
    L[o + 4] = T(1);
    L[o + 5] = T(0);
    L[o + 6] = L20;
    L[o + 7] = L21;
    L[o + 8] = T(1);
    D[u * 3 + 0] = D0;
    D[u * 3 + 1] = D1;
    D[u * 3 + 2] = D2;
}

template <typename T>
__global__ void
ldlt_apply_k3(const T* __restrict__ C, long long s0, long long s1, const int* __restrict__ x,
              const int* __restrict__ y, const int* __restrict__ Z, const int* __restrict__ inv,
              const T* __restrict__ L, const T* __restrict__ D, T* __restrict__ out, int B, T eps) {
    const int b = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    if (b >= B) {
        return;
    }
    const int u = inv[b];
    const int xi = x[b];
    const int yi = y[b];
    const int z0 = Z[u * 3];
    const int z1 = Z[u * 3 + 1];
    const int z2 = Z[u * 3 + 2];
    const int o = u * 9;
    const T L10 = L[o + 3];
    const T L20 = L[o + 6];
    const T L21 = L[o + 7];
    const T D0 = D[u * 3];
    const T D1 = D[u * 3 + 1];
    const T D2 = D[u * 3 + 2];
    const T ax0 = c_at(C, z0, xi, s0, s1);
    const T ax1 = c_at(C, z1, xi, s0, s1);
    const T ax2 = c_at(C, z2, xi, s0, s1);
    const T ay0 = c_at(C, z0, yi, s0, s1);
    const T ay1 = c_at(C, z1, yi, s0, s1);
    const T ay2 = c_at(C, z2, yi, s0, s1);
    const T axx = c_at(C, xi, xi, s0, s1);
    const T ayy = c_at(C, yi, yi, s0, s1);
    const T axy = c_at(C, xi, yi, s0, s1);
    const T ux0 = ax0;
    const T ux1 = ax1 - L10 * ux0;
    const T ux2 = ax2 - L20 * ux0 - L21 * ux1;
    const T uy0 = ay0;
    const T uy1 = ay1 - L10 * uy0;
    const T uy2 = ay2 - L20 * uy0 - L21 * uy1;
    const T inv0 = T(1) / (D0 + eps);
    const T inv1 = T(1) / (D1 + eps);
    const T inv2 = T(1) / (D2 + eps);
    const T qxy = axy - ux0 * uy0 * inv0 - ux1 * uy1 * inv1 - ux2 * uy2 * inv2;
    const T qxx = axx - ux0 * ux0 * inv0 - ux1 * ux1 * inv1 - ux2 * ux2 * inv2;
    const T qyy = ayy - uy0 * uy0 * inv0 - uy1 * uy1 * inv1 - uy2 * uy2 * inv2;
    out[b] = r_from_dets(qxy, qxx, qyy);
}

template <typename T, int K_MAX, int TPT>
__global__ void ldlt_factor_generic(const T* __restrict__ C, long long s0, long long s1,
                                    const int* __restrict__ Z, T* __restrict__ Lout,
                                    T* __restrict__ Dout, int U, int k) {
    static_assert(K_MAX >= 4 && K_MAX <= 96, "K_MAX in [4, 96]");
    static_assert(TPT >= 1 && TPT <= 32 && (32 % TPT) == 0, "TPT must divide 32");
    const int tests_per_cta = static_cast<int>(blockDim.x) / TPT;
    const int local = static_cast<int>(threadIdx.x) / TPT;
    const int t = static_cast<int>(threadIdx.x) % TPT;
    const int u = static_cast<int>(blockIdx.x) * tests_per_cta + local;
    const bool live = u < U;

    extern __shared__ unsigned char raw_factor[];
    T* smem_T = reinterpret_cast<T*>(raw_factor);
    const int t_stride = k * k + k;
    T* L = smem_T + static_cast<int>(local) * t_stride;
    T* D = L + k * k;
    int* smem_i = reinterpret_cast<int*>(smem_T + tests_per_cta * t_stride);
    int* z = smem_i + static_cast<int>(local) * k;

    if (live) {
        for (int i = t; i < k; i += TPT) {
            z[i] = Z[u * k + i];
        }
    }
    __syncwarp();

    if (live) {
        for (int idx = t; idx < k * k; idx += TPT) {
            const int i = idx / k;
            const int j = idx - i * k;
            L[idx] = c_at(C, z[i], z[j], s0, s1);
        }
    }
    __syncwarp();

    for (int j = 0; j < k; ++j) {
        if (live && t == 0) {
            T acc = L[j * k + j];
            for (int p = 0; p < j; ++p) {
                const T lp = L[j * k + p];
                acc -= lp * lp * D[p];
            }
            D[j] = acc;
            L[j * k + j] = T(1);
        }
        __syncwarp();
        if (live) {
            const T acc = D[j];
            for (int i = j + 1 + t; i < k; i += TPT) {
                T s = L[i * k + j];
                for (int p = 0; p < j; ++p) {
                    s -= L[i * k + p] * L[j * k + p] * D[p];
                }
                L[i * k + j] = (acc != T(0)) ? s / acc : T(0);
            }
        }
        __syncwarp();
    }

    if (t == 0 && live) {
        const int lo = u * k * k;
        for (int i = 0; i < k * k; ++i) {
            Lout[lo + i] = L[i];
        }
        const int d0 = u * k;
        for (int i = 0; i < k; ++i) {
            Dout[d0 + i] = D[i];
        }
    }
}

template <typename T, int K_MAX, int TPT>
__global__ void ldlt_apply_generic(const T* __restrict__ C, long long s0, long long s1,
                                   const int* __restrict__ x, const int* __restrict__ y,
                                   const int* __restrict__ Z, const int* __restrict__ inv,
                                   const T* __restrict__ Lglob, const T* __restrict__ Dglob,
                                   T* __restrict__ out, int B, int k, T eps) {
    static_assert(K_MAX >= 4 && K_MAX <= 96, "K_MAX in [4, 96]");
    static_assert(TPT >= 1 && TPT <= 32 && (32 % TPT) == 0, "TPT must divide 32");
    const int tests_per_cta = static_cast<int>(blockDim.x) / TPT;
    const int local = static_cast<int>(threadIdx.x) / TPT;
    const int t = static_cast<int>(threadIdx.x) % TPT;
    const int b = static_cast<int>(blockIdx.x) * tests_per_cta + local;
    const bool live = b < B;

    extern __shared__ unsigned char raw_apply[];
    T* smem_T = reinterpret_cast<T*>(raw_apply);
    const int t_stride = k * k + 3 * k;
    T* L = smem_T + static_cast<int>(local) * t_stride;
    T* D = L + k * k;
    T* ux = D + k;
    T* uy = ux + k;

    if (live) {
        const int u = inv[b];
        const int lo = u * k * k;
        const int d0 = u * k;
        for (int idx = t; idx < k * k; idx += TPT) {
            L[idx] = Lglob[lo + idx];
        }
        for (int i = t; i < k; i += TPT) {
            D[i] = Dglob[d0 + i];
        }
        const int xi = x[b];
        const int yi = y[b];
        for (int i = t; i < k; i += TPT) {
            const int zi = Z[u * k + i];
            ux[i] = c_at(C, zi, xi, s0, s1);
            uy[i] = c_at(C, zi, yi, s0, s1);
        }
    }
    __syncwarp();

    if (t == 0 && live) {
        solve_unit_lower(L, ux, k);
        solve_unit_lower(L, uy, k);
        const int xi = x[b];
        const int yi = y[b];
        const T axx = c_at(C, xi, xi, s0, s1);
        const T ayy = c_at(C, yi, yi, s0, s1);
        const T axy = c_at(C, xi, yi, s0, s1);
        out[b] = r_from_ux_uy(axx, ayy, axy, ux, uy, D, k, eps);
    }
}

template <typename T, int TPT>
void launch_factor_generic(cudaStream_t stream, const T* C, long long s0, long long s1,
                           const int* Z, T* L, T* D, int U, int k) {
    const size_t t_stride = static_cast<size_t>(k * k + k);
    const int tpc = tests_per_cta_fit(k, TPT, sizeof(T), t_stride);
    const int block = TPT * tpc;
    const int blocks = (U + tpc - 1) / tpc;
    const size_t smem = factor_smem_bytes(k, tpc, sizeof(T));
    auto* fn = ldlt_factor_generic<T, kMaxCond, TPT>;
    set_dyn_smem(fn, smem);
    fn<<<blocks, block, smem, stream>>>(C, s0, s1, Z, L, D, U, k);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int TPT>
void launch_apply_generic(cudaStream_t stream, const T* C, long long s0, long long s1, const int* x,
                          const int* y, const int* Z, const int* inv, const T* L, const T* D,
                          T* out, int B, int k, T eps) {
    const size_t t_stride = static_cast<size_t>(k * k + 3 * k);
    const int tpc = tests_per_cta_fit(k, TPT, sizeof(T), t_stride);
    const int block = TPT * tpc;
    const int blocks = (B + tpc - 1) / tpc;
    const size_t smem = generic_smem_bytes(k, tpc, sizeof(T));
    auto* fn = ldlt_apply_generic<T, kMaxCond, TPT>;
    set_dyn_smem(fn, smem);
    fn<<<blocks, block, smem, stream>>>(C, s0, s1, x, y, Z, inv, L, D, out, B, k, eps);
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void dispatch_factor(cudaStream_t stream, const T* C, long long s0, long long s1, const int* Z,
                     T* L, T* D, int U, int k) {
    const int tpt = tpt_for_k(k);
    const int blocks1 = (U + kBlock - 1) / kBlock;
    switch (k) {
    case 2:
        ldlt_factor_k2<T><<<blocks1, kBlock, 0, stream>>>(C, s0, s1, Z, L, D, U);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    case 3:
        ldlt_factor_k3<T><<<blocks1, kBlock, 0, stream>>>(C, s0, s1, Z, L, D, U);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    default:
        break;
    }
    switch (tpt) {
    case 16:
        launch_factor_generic<T, 16>(stream, C, s0, s1, Z, L, D, U, k);
        break;
    case 32:
        launch_factor_generic<T, 32>(stream, C, s0, s1, Z, L, D, U, k);
        break;
    default:
        launch_factor_generic<T, 8>(stream, C, s0, s1, Z, L, D, U, k);
        break;
    }
}

template <typename T>
void dispatch_apply(cudaStream_t stream, const T* C, long long s0, long long s1, const int* x,
                    const int* y, const int* Z, const int* inv, const T* L, const T* D, T* out,
                    int B, int k, T eps) {
    const int tpt = tpt_for_k(k);
    const int blocks1 = (B + kBlock - 1) / kBlock;
    switch (k) {
    case 2:
        ldlt_apply_k2<T>
            <<<blocks1, kBlock, 0, stream>>>(C, s0, s1, x, y, Z, inv, L, D, out, B, eps);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    case 3:
        ldlt_apply_k3<T>
            <<<blocks1, kBlock, 0, stream>>>(C, s0, s1, x, y, Z, inv, L, D, out, B, eps);
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    default:
        break;
    }
    switch (tpt) {
    case 16:
        launch_apply_generic<T, 16>(stream, C, s0, s1, x, y, Z, inv, L, D, out, B, k, eps);
        break;
    case 32:
        launch_apply_generic<T, 32>(stream, C, s0, s1, x, y, Z, inv, L, D, out, B, k, eps);
        break;
    default:
        launch_apply_generic<T, 8>(stream, C, s0, s1, x, y, Z, inv, L, D, out, B, k, eps);
        break;
    }
}

template <typename T> void touch_ldlt_t() {
    touch_cuda_kernel(fisher_z_k0<T, 1>);
    touch_cuda_kernel(fisher_z_k1<T, 1>);
    touch_cuda_kernel(fisher_z_k2<T, 1>);
    touch_cuda_kernel(fisher_z_k3<T, 1>);
    touch_cuda_kernel(fisher_z_ldlt_generic<T, kMaxCond, 8>);
    touch_cuda_kernel(fisher_z_ldlt_generic<T, kMaxCond, 16>);
    touch_cuda_kernel(fisher_z_ldlt_generic<T, kMaxCond, 32>);
    touch_cuda_kernel(ldlt_factor_k2<T>);
    touch_cuda_kernel(ldlt_factor_k3<T>);
    touch_cuda_kernel(ldlt_apply_k2<T>);
    touch_cuda_kernel(ldlt_apply_k3<T>);
    touch_cuda_kernel(ldlt_factor_generic<T, kMaxCond, 8>);
    touch_cuda_kernel(ldlt_factor_generic<T, kMaxCond, 16>);
    touch_cuda_kernel(ldlt_factor_generic<T, kMaxCond, 32>);
    touch_cuda_kernel(ldlt_apply_generic<T, kMaxCond, 8>);
    touch_cuda_kernel(ldlt_apply_generic<T, kMaxCond, 16>);
    touch_cuda_kernel(ldlt_apply_generic<T, kMaxCond, 32>);
}

void touch_all_ldlt() {
#if WHYDRA_ENABLE_FLOAT32
    touch_ldlt_t<float>();
#endif
    touch_ldlt_t<double>();
}

} // namespace

void fisher_z_ldlt_preload() {
    cuda_preload_context();
    touch_all_ldlt();
}

void fisher_z_ldlt_batch(torch::Tensor C, torch::Tensor x, torch::Tensor y, torch::Tensor Z,
                         torch::Tensor out, double eps) {
    // Per-test LDLᵀ. k≤3: CTA 256, TPT=1. k≥4: TPT 8/16/32, CTA shrinks
    // when dynamic smem of the k×k factor would exceed opt-in.
    TORCH_CHECK(C.is_cuda() && x.is_cuda() && y.is_cuda() && Z.is_cuda() && out.is_cuda(),
                "fisher_z_ldlt_batch tensors must be CUDA");
    TORCH_CHECK(C.dim() == 2 && C.size(0) == C.size(1), "C must be square");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && Z.is_contiguous() && out.is_contiguous(),
                "x, y, Z, out must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kInt && y.scalar_type() == at::kInt, "x, y must be int32");
    TORCH_CHECK(Z.scalar_type() == at::kInt, "Z must be int32");
    TORCH_CHECK(C.scalar_type() == out.scalar_type(), "C/out dtype mismatch");
#if WHYDRA_ENABLE_FLOAT32
    TORCH_CHECK(C.scalar_type() == at::kDouble || C.scalar_type() == at::kFloat,
                "fisher_z_ldlt_batch supports float64 and float32 only");
#else
    TORCH_CHECK(C.scalar_type() == at::kDouble, "fisher_z_ldlt_batch supports float64 only");
#endif
    const int64_t B = x.numel();
    TORCH_CHECK(y.numel() == B && out.numel() == B, "x, y, out batch sizes must match");
    TORCH_CHECK(Z.dim() == 2 && Z.size(0) == B, "Z must be (B, k)");
    const int k = static_cast<int>(Z.size(1));
    TORCH_CHECK(k >= 0 && k <= kMaxCond, "k out of range (0..96)");
    if (B == 0) {
        return;
    }
    const int64_t s0 = C.stride(0);
    const int64_t s1 = C.stride(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    WHYDRA_DISPATCH_FLOATING(C.scalar_type(), "fisher_z_ldlt_batch", [&] {
        dispatch_ldlt<scalar_t>(stream, C.data_ptr<scalar_t>(), s0, s1, x.data_ptr<int>(),
                                y.data_ptr<int>(), Z.data_ptr<int>(), out.data_ptr<scalar_t>(),
                                static_cast<int>(B), k, static_cast<scalar_t>(eps));
    });
}

int fisher_z_ldlt_k_max() {
    // Largest k whose generic-kernel shared-memory footprint (one test:
    // (k*k + 3k) doubles for the k×k tile + k int indices) fits the device
    // opt-in dynamic smem.  Compile-time kMaxCond=96 needs ~72 KB, which
    // datacenter GPUs (A40, 99 KB opt-in) allow but consumer parts
    // (e.g. RTX 2080 Ti, sm_75) cap at 64 KB — k=89 needs 65 860 bytes.
    // On those devices the returned cap makes the Python layers fall back
    // to the host combinadic wavefront for larger conditioning sets.
    static int cached = -1;
    if (cached < 0) {
        const int cap = optin_dyn_smem();
        int k = kMaxCond;
        while (k > 4 && generic_smem_bytes(k, 1, sizeof(double)) > static_cast<size_t>(cap)) {
            --k;
        }
        cached = k;
    }
    return cached;
}

int fisher_z_ldlt_tpt(int k) { return tpt_for_k(k); }

void fisher_z_ldlt_factor(torch::Tensor C, torch::Tensor Z, torch::Tensor L, torch::Tensor D,
                          double eps) {
    (void)eps;
    TORCH_CHECK(C.is_cuda() && Z.is_cuda() && L.is_cuda() && D.is_cuda(),
                "fisher_z_ldlt_factor tensors must be CUDA");
    TORCH_CHECK(C.dim() == 2 && C.size(0) == C.size(1), "C must be square");
    TORCH_CHECK(Z.is_contiguous() && L.is_contiguous() && D.is_contiguous(),
                "Z, L, D must be contiguous");
    TORCH_CHECK(Z.scalar_type() == at::kInt, "Z must be int32");
    TORCH_CHECK(C.scalar_type() == L.scalar_type() && C.scalar_type() == D.scalar_type(),
                "C/L/D dtype mismatch");
#if WHYDRA_ENABLE_FLOAT32
    TORCH_CHECK(C.scalar_type() == at::kDouble || C.scalar_type() == at::kFloat,
                "fisher_z_ldlt_factor supports float64 and float32 only");
#else
    TORCH_CHECK(C.scalar_type() == at::kDouble, "fisher_z_ldlt_factor supports float64 only");
#endif
    TORCH_CHECK(Z.dim() == 2, "Z must be (U, k)");
    const int64_t U = Z.size(0);
    const int k = static_cast<int>(Z.size(1));
    TORCH_CHECK(k >= 2 && k <= kMaxCond, "k out of range (2..96) for unique-S factor");
    TORCH_CHECK(L.sizes() == torch::IntArrayRef({U, k, k}), "L must be (U, k, k)");
    TORCH_CHECK(D.sizes() == torch::IntArrayRef({U, k}), "D must be (U, k)");
    if (U == 0) {
        return;
    }
    const int64_t s0 = C.stride(0);
    const int64_t s1 = C.stride(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    WHYDRA_DISPATCH_FLOATING(C.scalar_type(), "fisher_z_ldlt_factor", [&] {
        dispatch_factor<scalar_t>(stream, C.data_ptr<scalar_t>(), s0, s1, Z.data_ptr<int>(),
                                  L.data_ptr<scalar_t>(), D.data_ptr<scalar_t>(),
                                  static_cast<int>(U), k);
    });
}

void fisher_z_ldlt_apply(torch::Tensor C, torch::Tensor x, torch::Tensor y, torch::Tensor Z,
                         torch::Tensor inv, torch::Tensor L, torch::Tensor D, torch::Tensor out,
                         double eps) {
    TORCH_CHECK(C.is_cuda() && x.is_cuda() && y.is_cuda() && Z.is_cuda() && inv.is_cuda() &&
                    L.is_cuda() && D.is_cuda() && out.is_cuda(),
                "fisher_z_ldlt_apply tensors must be CUDA");
    TORCH_CHECK(C.dim() == 2 && C.size(0) == C.size(1), "C must be square");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && Z.is_contiguous() &&
                    inv.is_contiguous() && L.is_contiguous() && D.is_contiguous() &&
                    out.is_contiguous(),
                "x, y, Z, inv, L, D, out must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kInt && y.scalar_type() == at::kInt &&
                    Z.scalar_type() == at::kInt && inv.scalar_type() == at::kInt,
                "x, y, Z, inv must be int32");
    TORCH_CHECK(C.scalar_type() == out.scalar_type() && C.scalar_type() == L.scalar_type() &&
                    C.scalar_type() == D.scalar_type(),
                "C/L/D/out dtype mismatch");
#if WHYDRA_ENABLE_FLOAT32
    TORCH_CHECK(C.scalar_type() == at::kDouble || C.scalar_type() == at::kFloat,
                "fisher_z_ldlt_apply supports float64 and float32 only");
#else
    TORCH_CHECK(C.scalar_type() == at::kDouble, "fisher_z_ldlt_apply supports float64 only");
#endif
    const int64_t B = x.numel();
    TORCH_CHECK(y.numel() == B && out.numel() == B && inv.numel() == B,
                "x, y, inv, out batch sizes must match");
    TORCH_CHECK(Z.dim() == 2, "Z must be (U, k)");
    const int64_t U = Z.size(0);
    const int k = static_cast<int>(Z.size(1));
    TORCH_CHECK(k >= 2 && k <= kMaxCond, "k out of range (2..96) for unique-S apply");
    TORCH_CHECK(L.sizes() == torch::IntArrayRef({U, k, k}), "L must be (U, k, k)");
    TORCH_CHECK(D.sizes() == torch::IntArrayRef({U, k}), "D must be (U, k)");
    if (B == 0) {
        return;
    }
    const int64_t s0 = C.stride(0);
    const int64_t s1 = C.stride(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    WHYDRA_DISPATCH_FLOATING(C.scalar_type(), "fisher_z_ldlt_apply", [&] {
        dispatch_apply<scalar_t>(
            stream, C.data_ptr<scalar_t>(), s0, s1, x.data_ptr<int>(), y.data_ptr<int>(),
            Z.data_ptr<int>(), inv.data_ptr<int>(), L.data_ptr<scalar_t>(), D.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), static_cast<int>(B), k, static_cast<scalar_t>(eps));
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ldlt_batch", &fisher_z_ldlt_batch);
    m.def("ldlt_factor", &fisher_z_ldlt_factor);
    m.def("ldlt_apply", &fisher_z_ldlt_apply);
    m.def("k_max", &fisher_z_ldlt_k_max);
    m.def("tpt_for_k", &fisher_z_ldlt_tpt);
    m.def("preload", &fisher_z_ldlt_preload);
}
