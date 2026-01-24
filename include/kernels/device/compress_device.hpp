/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <hip/hip_runtime.h>

#include "core/detail/casting.hpp"

namespace Kernels {
namespace Device {
template <typename InWrapper, typename OutWrapper>
__global__ void diff_flags(InWrapper input, OutWrapper output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= input.width()) {
        return;
    }

    if (x == 0) {
        output.at(0, 0, x, 0) = 1;
    } else {
        output.at(0, 0, x, 0) = (input.at(0, 0, x, 0) != input.at(0, 0, x - 1, 0));
    }
}

template <typename InWrapper, typename OutWrapper>
__global__ void rl_index(InWrapper prefix_sum, OutWrapper rl_indexes, int *total_count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= prefix_sum.width()) {
        return;
    }

    if (x == (prefix_sum.width() - 1)) {
        rl_indexes.at(0, 0, prefix_sum.at(0, 0, x, 0), 0) = x + 1;
        *total_count = prefix_sum.at(0, 0, x, 0);
    }

    if (x == 0) {
        rl_indexes.at(0, 0, 0, 0) = 0;
    } else if (prefix_sum.at(0, 0, x, 0) != prefix_sum.at(0, 0, x - 1, 0)) {
        rl_indexes.at(0, 0, prefix_sum.at(0, 0, x, 0) - 1, 0) = x;
    }
}

template <typename InWrapper, typename rlIndexWrapper, typename OutWrapper>
__global__ void rl_code(InWrapper input, rlIndexWrapper rl_indexes, int *total_count, OutWrapper output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = *total_count;
    if (x >= n) {
        return;
    }

    // value
    output.at(0, 0, 2 * x, 0) = input.at(0, 0, rl_indexes.at(0, 0, x, 0), 0);
    // run-length
    output.at(0, 0, 2 * x + 1, 0) = rl_indexes.at(0, 0, x + 1, 0) - rl_indexes.at(0, 0, x, 0);
}
}  // namespace Device
}  // namespace Kernels