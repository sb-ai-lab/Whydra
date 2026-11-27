// Shared validation at the pybind boundary of the CUDA kernels.
#pragma once

#include <torch/extension.h>

// Validate the device CSR sepset contract used by udag2pag / meek / pdsep.
//
// ``ranges`` is ``(p, p, 2)`` int64 and ``S`` is a 1-D int64 member array.
// ``(-1, -1)`` means "no separator"; ``(0, 0)`` is an empty separator; every
// other entry must satisfy ``0 <= start <= end <= S.numel()``.
inline void require_sepset_csr_cuda(const torch::Tensor& S, const torch::Tensor& ranges, int p,
                                    const char* stage) {
    const bool s_defined = S.defined();
    const bool r_defined = ranges.defined();
    TORCH_CHECK(s_defined == r_defined, stage, ": S and sepset ranges must be provided together");
    if (!r_defined) {
        return;
    }
    TORCH_CHECK(S.is_cuda(), stage, ": S must be CUDA");
    TORCH_CHECK(S.scalar_type() == at::kLong, stage, ": S must be int64");
    TORCH_CHECK(S.is_contiguous(), stage, ": S must be contiguous");
    TORCH_CHECK(S.dim() == 1, stage, ": S must be 1-D, got dim ", S.dim());
    TORCH_CHECK(ranges.is_cuda(), stage, ": sepset ranges must be CUDA");
    TORCH_CHECK(ranges.scalar_type() == at::kLong, stage, ": sepset ranges must be int64");
    TORCH_CHECK(ranges.is_contiguous(), stage, ": sepset ranges must be contiguous");
    TORCH_CHECK(ranges.dim() == 3, stage, ": sepset ranges must have shape (p,p,2), got dim ",
                ranges.dim());
    TORCH_CHECK(ranges.size(0) == p && ranges.size(1) == p && ranges.size(2) == 2, stage,
                ": sepset ranges must have shape (", p, ",", p, ",2), got (", ranges.size(0), ",",
                ranges.size(1), ",", ranges.size(2), ")");
    if (p == 0) {
        return;
    }

    auto starts = ranges.select(2, 0);
    auto ends = ranges.select(2, 1);
    const int64_t min_start = starts.min().item<int64_t>();
    const int64_t min_end = ends.min().item<int64_t>();
    const int64_t max_end = ends.max().item<int64_t>();
    TORCH_CHECK(min_start >= -1, stage, ": sepset range start must be >= -1, got ", min_start);
    TORCH_CHECK(min_end >= -1, stage, ": sepset range end must be >= -1, got ", min_end);
    TORCH_CHECK(max_end <= S.numel(), stage, ": sepset ranges point past S (max end ", max_end,
                " > S.numel() ", S.numel(), ")");
    TORCH_CHECK(!((starts == -1) & (ends != -1)).any().item<bool>(), stage,
                ": a missing separator must be stored as (-1,-1)");
    TORCH_CHECK(!((starts >= 0) & (ends < starts)).any().item<bool>(), stage,
                ": sepset range end must not precede start");
}
