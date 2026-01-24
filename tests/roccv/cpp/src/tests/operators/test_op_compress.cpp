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

#include <hipcub/device/device_scan.hpp>
//#include <core/detail/casting.hpp>
//#include <core/detail/type_traits.hpp>
//#include <core/wrappers/image_wrapper.hpp>
#include <op_compress.hpp>
#include <chrono> // Jefftest
#include <hipcub/device/device_scan.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

#define HIP_API_CALL( call )                                                                                  \
    do {                                                                                                      \
        hipError_t hip_status = call;                                                                         \
        if (hip_status != hipSuccess) {                                                                       \
            const char *sz_err_name = NULL;                                                                     \
            sz_err_name = hipGetErrorName(hip_status);                                                          \
            std::ostringstream error_log;                                                                     \
            error_log << "hip API error " << sz_err_name ;                                                      \
            throw(error_log.str());                   \
        }                                                                                                     \
    }                                                                                                         \
    while (0)

namespace {
void get_prefix_sum(hipStream_t stream, void *input, void *output, int num_items) {
    void* d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    int* d_in = static_cast<int*>(input);
    int* d_out = static_cast<int*>(output);

    HIP_API_CALL(hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
    HIP_API_CALL(hipMalloc(&d_temp_storage, temp_storage_bytes));
    HIP_API_CALL(hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
}

void RunLengthEncode(std::vector<uint32_t>& in_buffer, std::vector<uint32_t>& out_buffer, uint32_t *code_count) {
    int in_buf_size = in_buffer.size();
    int out_buf_size = out_buffer.size();
    Tensor input(1, {in_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    Tensor diff_flag(1, {in_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    Tensor prefix_sum(1, {in_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    Tensor rl_indices(1, {in_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    Tensor output(1, {out_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    int *num_pair;
    HIP_API_CALL(hipMalloc(&num_pair, sizeof(int)));

    CopyVectorIntoTensor(input, in_buffer);
    CopyVectorIntoTensor(output, out_buffer);
    
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    Compress op;

    for (int i = 0; i < 1; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        op.get_diff_flags(stream, input, diff_flag);

        auto diff_flag_data = diff_flag.exportData<TensorDataStrided>();
        void *diff_flag_ptr = diff_flag_data.basePtr();
        auto prefix_sum_data = prefix_sum.exportData<TensorDataStrided>();
        void *prefix_sum_ptr = prefix_sum_data.basePtr();
        get_prefix_sum(stream, diff_flag_ptr, prefix_sum_ptr, in_buf_size);

        op.get_rl_indexes(stream, prefix_sum, rl_indices, num_pair);

        op.get_rl_code(stream, input, rl_indices, num_pair, output);
        
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "<Profiling> hipStreamSynchronize() elapsed time for iteration " << i << ": " << elapsed << " ms" << std::endl;
        //float throughput =  in_buf_size * sizeof(in_buffer[0]) * 3 / 1024.0 / 1024.0 / 1024.0 * 1000.0 / elapsed;
        //std::cout << "<Profiling> Kernel throughput based on C++ Chrono lib for iteration " << i << ": " << throughput << " GB/s" << std::endl;

    }
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    std::vector<uint32_t> diff_flag_array(diff_flag.shape().size());
    CopyTensorIntoVector(diff_flag_array, diff_flag);
    std::cout << "Diff flag array: ";
    for (int i = 0; i < diff_flag_array.size(); i++) {
        std::cout << diff_flag_array[i] << " ";
    }
    std::cout << std::endl;

    std::vector<uint32_t> prefix_sum_array(prefix_sum.shape().size());
    CopyTensorIntoVector(prefix_sum_array, prefix_sum);
    std::cout << "Prefix sum array: ";
    for (int i = 0; i < prefix_sum_array.size(); i++) {
        std::cout << prefix_sum_array[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "RL pair count: " << *num_pair << std::endl;
    std::vector<uint32_t> rl_index_array(rl_indices.shape().size());
    CopyTensorIntoVector(rl_index_array, rl_indices);
    std::cout << "RL index array: ";
    for (int i = 0; i < *num_pair + 1; i++) {
        std::cout << rl_index_array[i] << " ";
    }
    std::cout << std::endl;

    std::vector<uint32_t> output_array(output.shape().size());
    CopyTensorIntoVector(output_array, output);
    std::cout << "Output array: ";
    for (int i = 0; i < *num_pair; i++) {
        std::cout << "(" << output_array[i * 2] << ", " << output_array[i * 2 + 1] << "), ";
    }
    std::cout << std::endl;
}
}  // namespace

int main(int argc, char **argv) {
    printf("Running compress test ...\n");
    std::vector<uint32_t> in_buffer = {1,2,3,6,6,6,5,5};
    uint32_t in_count = in_buffer.size(); // sizeof(in_buffer) / sizeof(uint32_t);
    printf("Input size = %d\n", in_count);
    uint32_t max_out_count = in_count * 2;
    std::vector<uint32_t> out_buffer(max_out_count, 0);
    uint32_t code_count = 0;

    //TestCorrectness<uchar3, uchar1, uchar3>(32, 1024, 1024, FMT_RGB8, FMT_U8, FMT_RGB8, eDeviceType::GPU);

    RunLengthEncode(in_buffer, out_buffer, &code_count);

}