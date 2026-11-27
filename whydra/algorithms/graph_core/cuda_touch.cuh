#pragma once

// Load a __global__ into the CUDA context without launching it.
// The prebuilt CUDAExtension already nvcc-compiled the cubin;
// CUDA_MODULE_LOADING=LAZY still defers each instantiation until first use.
// cudaFuncGetAttributes is that first use. Grid size p is runtime and is
// not specialized.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename Kernel> inline void touch_cuda_kernel(Kernel kernel) {
    cudaFuncAttributes attr{};
    const cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
    TORCH_CHECK(err == cudaSuccess, "cudaFuncGetAttributes: ", cudaGetErrorString(err));
}

inline void cuda_preload_context() {
    TORCH_CHECK(at::cuda::is_available(), "CUDA required to preload kernels");
    (void)at::cuda::getCurrentCUDAStream();
}
