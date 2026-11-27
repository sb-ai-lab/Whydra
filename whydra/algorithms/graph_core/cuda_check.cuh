#pragma once

#include <cuda_runtime.h>

// Local replacement for C10_CUDA_KERNEL_LAUNCH_CHECK() from
// <c10/cuda/CUDAException.h>: torch >= 2.10 removed
// c10::cuda::CUDAErrorLogCapture from the exported ABI, so the c10 macro no
// longer links against newer libtorch (undefined symbol on import). Same
// semantics — check the last launch error and raise — using only the stable
// TORCH_CHECK / CUDA runtime surface, so kernels build on old and new torch
// alike.
#define WHYDRA_CUDA_KERNEL_LAUNCH_CHECK()                                                          \
    do {                                                                                           \
        cudaError_t _whydra_cuda_err_ = cudaGetLastError();                                        \
        TORCH_CHECK(_whydra_cuda_err_ == cudaSuccess,                                              \
                    "CUDA kernel launch failed: ", cudaGetErrorString(_whydra_cuda_err_));         \
    } while (0)
