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

//#include <hipcub/device/device_scan.hpp>
#include "op_compress.hpp"

#include <functional>

#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/compress_device.hpp"

namespace roccv {
template <typename DataType>
void dispatch_diff_flag_datatype(hipStream_t stream, const Tensor& input, const Tensor& output) {
    ImageWrapper<DataType> inputWrapper(input);
    ImageWrapper<DataType> outputWrapper(output);

    // Jefftest 
    dim3 block(64);
    //dim3 block(1024);
    dim3 grid((inputWrapper.width() + block.x - 1) / block.x);
    Kernels::Device::diff_flags<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper);
}

void Compress::get_diff_flags(hipStream_t stream, Tensor& input, Tensor& diff_flag) {
    // clang-format off
    static const std::unordered_map<eDataType, std::function<void(hipStream_t, const Tensor&, const Tensor&)>>
        funcs = {
            {eDataType::DATA_TYPE_U32,  dispatch_diff_flag_datatype<uint32_t>}
        };
    // clang-format on
    auto func = funcs.at(input.dtype().etype());
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, diff_flag);
}

template <typename DataType>
void dispatch_rl_index_datatype(hipStream_t stream, const Tensor& input, const Tensor& output, int *total_count) {
    ImageWrapper<DataType> inputWrapper(input);
    ImageWrapper<DataType> outputWrapper(output);

    dim3 block(64);
    dim3 grid((inputWrapper.width() + block.x - 1) / block.x);
    Kernels::Device::rl_index<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, total_count);
}

void Compress::get_rl_indexes(hipStream_t stream, Tensor& prefix_sum, Tensor& rl_index, int *total_count) {
    // clang-format off
    static const std::unordered_map<eDataType, std::function<void(hipStream_t, const Tensor&, const Tensor&, int*)>>
        funcs = {
            {eDataType::DATA_TYPE_U32,  dispatch_rl_index_datatype<uint32_t>}
        };
    // clang-format on
    auto func = funcs.at(prefix_sum.dtype().etype());
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, prefix_sum, rl_index, total_count);
}

template <typename DataType>
void dispatch_rl_code_datatype(hipStream_t stream, const Tensor& input, const Tensor& rl_index, int *total_count, const Tensor& output) {
    ImageWrapper<DataType> inputWrapper(input);
    ImageWrapper<DataType> rlIndexWrapper(rl_index);
    ImageWrapper<DataType> outputWrapper(output);

    dim3 block(64);
    dim3 grid((inputWrapper.width() + block.x - 1) / block.x);
    Kernels::Device::rl_code<<<grid, block, 0, stream>>>(inputWrapper, rlIndexWrapper, total_count, outputWrapper);
}

void Compress::get_rl_code(hipStream_t stream, Tensor& input, Tensor& rl_index, int *total_count, Tensor& output) {
    // clang-format off
    static const std::unordered_map<eDataType, std::function<void(hipStream_t, const Tensor&, const Tensor&, int*, const Tensor&)>>
        funcs = {
            {eDataType::DATA_TYPE_U32,  dispatch_rl_code_datatype<uint32_t>}
        };
    // clang-format on    
    auto func = funcs.at(input.dtype().etype());
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, rl_index, total_count, output);
}

};  // namespace roccv