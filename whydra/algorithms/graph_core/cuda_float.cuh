#pragma once

// CI kernels (Fisher-Z LDLᵀ, fused ci_from_comb, prefix_pmax, first_hit,
// hits_from_r) compile float64 only by default. Re-enable float32 with
// WHYDRA_ENABLE_FLOAT32=1 at nvcc time (setup.py reads that env var).

#ifndef WHYDRA_ENABLE_FLOAT32
#define WHYDRA_ENABLE_FLOAT32 0
#endif

#include <ATen/Dispatch.h>

#if WHYDRA_ENABLE_FLOAT32
#define WHYDRA_DISPATCH_FLOATING(TYPE, NAME, ...)                                                  \
    AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__)
#else
#define WHYDRA_DISPATCH_FLOATING(TYPE, NAME, ...)                                                  \
    do {                                                                                           \
        TORCH_CHECK((TYPE) == at::kDouble,                                                         \
                    NAME ": float64 only (rebuild with WHYDRA_ENABLE_FLOAT32=1)");                 \
        using scalar_t = double;                                                                   \
        auto whydra_dispatch_fn = __VA_ARGS__;                                                     \
        whydra_dispatch_fn();                                                                      \
    } while (0)
#endif
