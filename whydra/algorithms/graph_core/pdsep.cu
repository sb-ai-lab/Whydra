// pdsep colliders + Possible-D-SEP reachability (qreach) on a dense amat.
//
// amat[i,j]: 0 none, 1 adjacent, 2 collider arrow into j.
// Sepsets are CSR: ranges[x,z]=(start,end) into S. Missing/empty: start<0 or end<=start.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cuda_check.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <tuple>
#include "cuda_touch.cuh"
#include "cuda_validate.cuh"

namespace {

__device__ __forceinline__ int8_t amat_at(const int8_t* amat, int p, int i, int j) {
    return amat[static_cast<long long>(i) * p + j];
}

__device__ __forceinline__ void amat_put(int8_t* amat, int p, int i, int j, int8_t v) {
    amat[static_cast<long long>(i) * p + j] = v;
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

// For each endpoint (u, center), look for a second unshielded neighbor.
// Only the owning lane writes this endpoint, after every witness has been read.
__global__ void mark_colliders_kernel(int8_t* amat, int p, const int64_t* S, const int64_t* ranges,
                                      const int32_t* unf, int U, int* mark) {
    for (int center = 0; center != p; ++center) {
        for (int u = threadIdx.x; u < p; u += blockDim.x) {
            const bool source = amat_at(amat, p, u, center) == 1;
            const bool neighbor = amat_at(amat, p, center, u) != 0;
            bool orient = false;
            if (source || neighbor) {
                for (int v = 0; v < p && !orient; ++v) {
                    if (u == v)
                        continue;
                    const bool uv =
                        source && amat_at(amat, p, center, v) && !amat_at(amat, p, u, v);
                    const bool vu =
                        neighbor && amat_at(amat, p, v, center) == 1 && !amat_at(amat, p, v, u);
                    if (uv || vu)
                        orient = !is_unf(unf, U, u, center, v) &&
                                 !in_sep(S, ranges, p, u, v, center) &&
                                 !in_sep(S, ranges, p, v, u, center);
                }
            }
            mark[u] = orient;
        }
        __syncthreads();
        for (int u = threadIdx.x; u < p; u += blockDim.x)
            if (mark[u])
                amat_put(amat, p, u, center, 2);
        __syncthreads();
    }
}

__device__ __forceinline__ bool used_get(const uint8_t* used, int p, int a, int b) {
    const long long bit = static_cast<long long>(a) * p + b;
    return (used[bit >> 3] >> (bit & 7)) & 1;
}

__device__ __forceinline__ void used_set(uint8_t* used, int p, int a, int b) {
    const long long bit = static_cast<long long>(a) * p + b;
    used[bit >> 3] |= static_cast<uint8_t>(1u << (bit & 7));
}

__global__ void degrees_kernel(const int8_t* amat, int p, int* deg) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= p) {
        return;
    }
    int d = 0;
    const long long r = static_cast<long long>(i) * p;
    for (int j = 0; j < p; ++j) {
        if (amat[r + j] != 0) {
            ++d;
        }
    }
    deg[i] = d;
}

__global__ void compact_kernel(const int8_t* amat, int p, int max_d, int* nbrs) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= p) {
        return;
    }
    int d = 0;
    const long long r = static_cast<long long>(i) * p;
    int* row = nbrs + static_cast<long long>(i) * max_d;
    for (int j = 0; j < p; ++j) {
        if (amat[r + j] != 0) {
            row[d++] = j;
        }
    }
}

// Reachability in the directed line graph. A state encodes the last edge
// of a walk. Depth-first expansion suffices because the public result is a mask.
__global__ void qreach_wave_kernel(const int8_t* amat, const int* nbrs, const int* deg, int p,
                                   int max_d, int x0, uint8_t* scratch, int stride,
                                   int64_t* workspace, int capacity, uint8_t* mask,
                                   uint8_t* overflow) {
    const int lane = threadIdx.x, root = x0 + lane;
    int64_t* stack = workspace + static_cast<int64_t>(lane) * capacity;
    uint8_t* seen = scratch + static_cast<int64_t>(lane) * stride;
    uint8_t* result = mask + static_cast<int64_t>(root) * p;
    int pending = 0;
    int states = 0;
    for (int i = 0; i < deg[root]; ++i) {
        const int v = nbrs[static_cast<int64_t>(root) * max_d + i];
        if (++states > capacity) {
            overflow[root] = 1;
            return;
        }
        stack[pending++] = static_cast<int64_t>(root) * p + v;
        result[v] = 1;
    }
    while (pending) {
        const int64_t edge = stack[--pending];
        const int u = edge / p, v = edge % p;
        for (int i = 0; i < deg[v]; ++i) {
            const int w = nbrs[static_cast<int64_t>(v) * max_d + i];
            if (w == u || w == root || used_get(seen, p, v, w))
                continue;
            const bool open_collider = amat_at(amat, p, u, v) == 2 && amat_at(amat, p, w, v) == 2;
            if (amat_at(amat, p, u, w) || open_collider) {
                if (++states > capacity) {
                    overflow[root] = 1;
                    return;
                }
                result[w] = 1;
                used_set(seen, p, v, w);
                stack[pending++] = static_cast<int64_t>(v) * p + w;
            }
        }
    }
}

} // namespace

void mark_colliders_cuda(torch::Tensor amat, torch::Tensor S, torch::Tensor ranges,
                         torch::Tensor unf) {
    TORCH_CHECK(amat.is_cuda() && amat.is_contiguous(), "amat must be contiguous CUDA");
    TORCH_CHECK(amat.dim() == 2 && amat.size(0) == amat.size(1), "amat must be square");
    TORCH_CHECK(amat.scalar_type() == at::kChar, "amat must be int8");
    const int p = static_cast<int>(amat.size(0));
    require_sepset_csr_cuda(S, ranges, p, "pdsep mark_colliders");
    TORCH_CHECK(unf.is_cuda() && unf.is_contiguous() && unf.scalar_type() == at::kInt, "unf");
    TORCH_CHECK(unf.dim() == 2 && unf.size(1) == 3, "unf (U,3)");
    if (p == 0) {
        return;
    }
    const int U = static_cast<int>(unf.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto mark = torch::zeros({p}, amat.options().dtype(at::kInt));
    const int t = 256;
    mark_colliders_kernel<<<1, t, 0, stream>>>(amat.data_ptr<int8_t>(), p, S.data_ptr<int64_t>(),
                                               ranges.data_ptr<int64_t>(), unf.data_ptr<int32_t>(),
                                               U, mark.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor> qreach_all_cuda(torch::Tensor amat) {
    TORCH_CHECK(amat.is_cuda() && amat.is_contiguous(), "amat must be contiguous CUDA");
    TORCH_CHECK(amat.dim() == 2 && amat.size(0) == amat.size(1), "amat must be square");
    TORCH_CHECK(amat.scalar_type() == at::kChar, "amat must be int8");
    const int p = static_cast<int>(amat.size(0));
    auto mask = torch::zeros({p, p}, amat.options().dtype(at::kByte));
    auto overflow = torch::zeros({p}, amat.options().dtype(at::kByte));
    if (p == 0) {
        return {mask, overflow};
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto deg = torch::empty({p}, amat.options().dtype(at::kInt));
    const int t = 256;
    const int g = (p + t - 1) / t;
    degrees_kernel<<<g, t, 0, stream>>>(amat.data_ptr<int8_t>(), p, deg.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    const int max_d = std::max(1, static_cast<int>(deg.max().item<int>()));
    auto nbrs = torch::empty({p, max_d}, amat.options().dtype(at::kInt));
    compact_kernel<<<g, t, 0, stream>>>(amat.data_ptr<int8_t>(), p, max_d, nbrs.data_ptr<int>());
    WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();

    const int T = std::min(p, 128);
    const int qmax = std::max(p * 16, 4096);
    const int used_stride = static_cast<int>((static_cast<long long>(p) * p + 7) / 8);
    auto used = torch::empty({T, used_stride}, amat.options().dtype(at::kByte));
    auto stack = torch::empty({T, qmax}, amat.options().dtype(at::kLong));
    for (int x0 = 0; x0 < p; x0 += T) {
        const int n = std::min(T, p - x0);
        used.zero_();
        qreach_wave_kernel<<<1, n, 0, stream>>>(
            amat.data_ptr<int8_t>(), nbrs.data_ptr<int>(), deg.data_ptr<int>(), p, max_d, x0,
            used.data_ptr<uint8_t>(), used_stride, stack.data_ptr<int64_t>(), qmax,
            mask.data_ptr<uint8_t>(), overflow.data_ptr<uint8_t>());
        WHYDRA_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {mask, overflow};
}

void pdsep_preload() {
    cuda_preload_context();
    touch_cuda_kernel(mark_colliders_kernel);
    touch_cuda_kernel(degrees_kernel);
    touch_cuda_kernel(compact_kernel);
    touch_cuda_kernel(qreach_wave_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mark_colliders", &mark_colliders_cuda);
    m.def("qreach_all", &qreach_all_cuda);
    m.def("preload", &pdsep_preload);
}
