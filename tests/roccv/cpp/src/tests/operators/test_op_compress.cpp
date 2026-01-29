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
#include <chrono>
#include <hipcub/device/device_scan.hpp>
#include <hipcub/device/device_run_length_encode.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {
void RunLengthEncodeGolden(uint32_t *input, uint32_t input_size, uint32_t *output, uint32_t* output_size) {
    int pair_index = 0;
    int value = input[0];
    int run_length = 1;

    for (int i = 1; i < input_size; i++) {
        if (input[i] != value) {
            output[pair_index * 2] = value;
            output[pair_index * 2 + 1] = run_length;
            pair_index++;

            value = input[i];
            run_length = 1;
        } else {
            run_length++;
        }
    }

    // Last pair
    output[pair_index * 2] = value;
    output[pair_index * 2 + 1] = run_length;
    pair_index++;

    *output_size = pair_index;
    // Jefftest
    /*std::cout << "Golden output array: ";
    for (int i = 0; i < *output_size; i++) {
        std::cout << "(" << output[i * 2] << ", " << output[i * 2 + 1] << "), ";
    }
    std::cout << std::endl;*/
}

void GetPrefixSum(hipStream_t stream, void *input, void *output, int num_items) {
    void* d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    int* d_in = static_cast<int*>(input);
    int* d_out = static_cast<int*>(output);

    HIP_VALIDATE_NO_ERRORS(hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_temp_storage, temp_storage_bytes));
    HIP_VALIDATE_NO_ERRORS(hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
}

void RunLengthEncode(std::vector<uint32_t>& in_buffer, std::vector<uint32_t>& out_buffer, uint32_t *code_count) {
    int in_buf_size = in_buffer.size();
    int out_buf_size = out_buffer.size();
    Tensor input(1, {in_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    Tensor diff_flag(1, {in_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    Tensor prefix_sum(1, {in_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    Tensor rl_indices(1, {in_buf_size + 1, 1}, FMT_U32, eDeviceType::GPU); // max possible size
    Tensor output(1, {out_buf_size, 1}, FMT_U32, eDeviceType::GPU);
    int *num_pair_d;
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&num_pair_d, sizeof(int)));

    CopyVectorIntoTensor(input, in_buffer);
    CopyVectorIntoTensor(output, out_buffer);
    
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    Compress op;

    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        op.get_diff_flags(stream, input, diff_flag);

        auto diff_flag_data = diff_flag.exportData<TensorDataStrided>();
        void *diff_flag_ptr = diff_flag_data.basePtr();
        auto prefix_sum_data = prefix_sum.exportData<TensorDataStrided>();
        void *prefix_sum_ptr = prefix_sum_data.basePtr();
        GetPrefixSum(stream, diff_flag_ptr, prefix_sum_ptr, in_buf_size);

        op.get_rl_indexes(stream, prefix_sum, rl_indices, num_pair_d);

        op.get_rl_code(stream, input, rl_indices, num_pair_d, output);
        
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "<Profiling> hipStreamSynchronize() elapsed time for iteration " << i << ": " << elapsed << " ms" << std::endl;
        float throughput =  in_buf_size * sizeof(in_buffer[0]) / 1000.0 / 1000.0 / 1000.0 * 1000.0 / elapsed;
        std::cout << "<Profiling> Kernel throughput based on C++ Chrono lib for iteration " << i << ": " << throughput << " GB/s" << std::endl;

    }

    HIP_VALIDATE_NO_ERRORS(hipMemcpy(code_count, num_pair_d, sizeof(int), hipMemcpyDeviceToHost));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    /*std::vector<uint32_t> diff_flag_array(diff_flag.shape().size());
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

    std::cout << "RL pair count: " << *code_count << std::endl;
    std::vector<uint32_t> rl_index_array(rl_indices.shape().size());
    CopyTensorIntoVector(rl_index_array, rl_indices);
    std::cout << "RL index array: ";
    for (int i = 0; i < *code_count + 1; i++) {
        std::cout << rl_index_array[i] << " ";
    }
    std::cout << std::endl;*/

    CopyTensorIntoVector(out_buffer, output);
    // Jefftest
    /*std::cout << "Output array: ";
    for (int i = 0; i < *code_count; i++) {
        std::cout << "(" << out_buffer[i * 2] << ", " << out_buffer[i * 2 + 1] << "), ";
    }
    std::cout << std::endl;*/
}

void RunLengthEncodeHipCUB(std::vector<uint32_t>& in_buffer, std::vector<uint32_t>& out_value_buffer, std::vector<uint32_t>& out_run_buffer, uint32_t *code_count) {
    int in_buf_size = in_buffer.size();

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    uint32_t* d_in;
    uint32_t* d_value_out;
    uint32_t* d_run_out;
    uint32_t* d_runs_count_output;

    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_in, in_buf_size * sizeof(uint32_t)));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(d_in, in_buffer.data(), in_buf_size * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_value_out, in_buf_size * sizeof(uint32_t)));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_run_out, in_buf_size * sizeof(uint32_t)));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_runs_count_output, sizeof(uint32_t)));

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_VALIDATE_NO_ERRORS(hipcub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes, d_in, d_value_out, d_run_out, d_runs_count_output, in_buf_size, stream));
        HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_VALIDATE_NO_ERRORS(hipcub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in, d_value_out, d_run_out, d_runs_count_output, in_buf_size, stream));
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "<Profiling> hipStreamSynchronize() elapsed time for iteration " << i << ": " << elapsed << " ms" << std::endl;
        float throughput =  in_buf_size * sizeof(in_buffer[0]) / 1000.0 / 1000.0 / 1000.0 * 1000.0 / elapsed;
        std::cout << "<Profiling> hipcub::DeviceRunLengthEncode::Encode throughput based on C++ Chrono lib for iteration " << i << ": " << throughput << " GB/s" << std::endl;
    }

    HIP_VALIDATE_NO_ERRORS(hipMemcpy(code_count, d_runs_count_output, sizeof(int), hipMemcpyDeviceToHost));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(out_value_buffer.data(), d_value_out, *code_count * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(out_run_buffer.data(), d_run_out, *code_count * sizeof(uint32_t), hipMemcpyDeviceToHost));

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));
}
}  // namespace

int main(int argc, char **argv) {
    printf("Running compress test ...\n");
    //std::vector<uint32_t> in_buffer = {1,2,3,6,6,6,5,5};

    // Create a vector and fill it with random data.
    uint32_t in_count = 1024 * 1024 * 100;
    std::vector<uint32_t> in_buffer(in_count);
    //FillVector(in_buffer);
    for (int i = 0; i < in_count; i++) {
        in_buffer[i] = i / 1;
    }

    printf("Input size = %d\n", in_count);
    uint32_t max_out_count = in_count * 2;
    std::vector<uint32_t> out_buffer(max_out_count, 0);
    uint32_t code_count = 0;
    std::vector<uint32_t> out_buffer_gold(max_out_count, 0);
    uint32_t code_count_gold = 0;

    // GPU
    RunLengthEncode(in_buffer, out_buffer, &code_count);
    std::cout << "Code pair count: " << code_count << std::endl;

    // CPU
    RunLengthEncodeGolden(in_buffer.data(), in_count, out_buffer_gold.data(), &code_count_gold);
    // Jefftest
    std::cout << "Gold code pair count: " << code_count_gold << std::endl;

    std::cout << "RLE compression ratio: " << static_cast<float>(in_count) / static_cast<float>(code_count) / 2.0f  << std::endl;

    // Compare
    out_buffer.resize(code_count * 2);
    out_buffer_gold.resize(code_count_gold * 2);
    CompareVectorsNear(out_buffer, out_buffer_gold, 0);

    // hipCUB path
    std::vector<uint32_t> out_value_buffer(in_count, 0);
    std::vector<uint32_t> out_run_buffer(in_count, 0);
    RunLengthEncodeHipCUB(in_buffer, out_value_buffer, out_run_buffer, &code_count);
    std::cout << "hipCUB code pair count: " << code_count << std::endl;
    // Combine value and run
    std::vector<uint32_t> out_buffer_2(code_count * 2, 0);
    for (int i = 0; i < code_count; i++) {
        out_buffer_2[i * 2] = out_value_buffer[i];
        out_buffer_2[i * 2 + 1] = out_run_buffer[i];
    }
    CompareVectorsNear(out_buffer_2, out_buffer_gold, 0);
}