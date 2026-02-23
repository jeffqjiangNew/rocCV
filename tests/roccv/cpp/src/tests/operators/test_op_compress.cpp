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
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_run_length_decode.hpp>
#include <hipcub/block/block_store.hpp>
#include "compress_util.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {
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
        GetPrefixSumHipCUB(stream, static_cast<uint32_t*>(diff_flag_ptr), static_cast<uint32_t*>(prefix_sum_ptr), in_buf_size);

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

        RunLengthEncodeHipCUB(stream, d_in, d_value_out, d_run_out, d_runs_count_output, in_buf_size);
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

#if 0 // Jefftest
void RunLengthDecodeHipCUB(std::vector<uint32_t>& in_value_buffer, std::vector<uint32_t>& in_run_buffer, std::vector<uint32_t>& out_buffer, uint32_t *out_buffer_size) {
    const uint32_t BlockSize = 32; // Jefftest 64;
    const uint32_t RunsPerThread = 2;
    const uint32_t DecodedItemsPerThread = 2;
    constexpr auto runs_per_block  = BlockSize * RunsPerThread;


    uint32_t *d_value_buf = nullptr;
    uint32_t *d_run_buf = nullptr;
    uint32_t *d_decoded_buf = nullptr;

    uint32_t in_code_size = in_value_buffer.size();
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_value_buf, in_code_size * sizeof(uint32_t)));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(d_value_buf, in_value_buffer.data(), in_code_size * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_run_buf, in_code_size * sizeof(uint32_t)));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(d_run_buf, in_run_buffer.data(), in_code_size * sizeof(uint32_t), hipMemcpyHostToDevice));

    // Use CPU to get decoded buffer size. Note this is just for testing purpose.
    uint32_t dec_buf_size = 0;
    for (int i = 0; i < in_code_size; i++) {
        dec_buf_size += in_run_buffer[i];
    }
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_decoded_buf, dec_buf_size * sizeof(uint32_t)));

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    // Jefftest block_run_length_decode_kernel<BlockSize, RunsPerThread, DecodedItemsPerThread><<<dim3(1), dim3(BlockSize), 0, stream>>>(d_value_buf, d_run_buf, d_decoded_buf);
    uint32_t num_blocks = (in_code_size + runs_per_block - 1) / runs_per_block;
    block_run_length_decode_kernel<BlockSize, RunsPerThread, DecodedItemsPerThread><<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_value_buf, d_run_buf, d_decoded_buf);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    out_buffer.resize(dec_buf_size);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(out_buffer.data(), d_decoded_buf, dec_buf_size * sizeof(uint32_t), hipMemcpyDeviceToHost));
    *out_buffer_size = dec_buf_size;

    // Jefftest
    for (int i = 0; i < dec_buf_size; i++) {
        std::cout << out_buffer[i] << " ";
    }
    std::cout << std::endl;
}
#endif

void RunLengthDecodeHipCUB_OffsetGPU(std::vector<uint32_t>& in_value_buffer, std::vector<uint32_t>& in_run_buffer, std::vector<uint32_t>& out_buffer, uint32_t *out_buffer_size) {
    const uint32_t BlockSize =  256; // 256; // Jefftest 64;
    const uint32_t RunsPerThread = 1; // 2;
    const uint32_t DecodedItemsPerThread = 1; // 2;
    constexpr auto runs_per_block  = BlockSize * RunsPerThread;

    uint32_t *d_value_buf = nullptr;
    uint32_t *d_offset_buf = nullptr;
    uint32_t *d_decoded_buf = nullptr;

    uint32_t in_code_size = in_value_buffer.size();

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_value_buf, in_value_buffer.size() * sizeof(in_value_buffer[0])));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(d_value_buf, in_value_buffer.data(), in_value_buffer.size() * sizeof(in_value_buffer[0]), hipMemcpyHostToDevice));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_offset_buf, in_run_buffer.size() * sizeof(in_run_buffer[0])));

    // Use GPU to get offsets
    uint32_t *d_run_buf = nullptr;
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_run_buf, in_run_buffer.size() * sizeof(in_run_buffer[0])));
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(d_run_buf, in_run_buffer.data(), in_run_buffer.size() * sizeof(in_run_buffer[0]), hipMemcpyHostToDevice));
    GetPrefixSumHipCUB(stream, d_run_buf, d_offset_buf, in_run_buffer.size());
    std::vector<uint32_t> presum_offset_buf(in_run_buffer.size());
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(presum_offset_buf.data(), d_offset_buf, presum_offset_buf.size() * sizeof(presum_offset_buf[0]), hipMemcpyDeviceToHost));
    // Jefftest
    /*for (int i = 0; i < presum_offset_buf.size(); i++) {
        std::cout << presum_offset_buf[i] << " ";
    }
    std::cout << std::endl;*/

    uint32_t dec_buf_size = presum_offset_buf.back();
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_decoded_buf, dec_buf_size * sizeof(uint32_t)));
    HIP_VALIDATE_NO_ERRORS(hipMemset(d_offset_buf, 0, in_run_buffer.size() * sizeof(in_run_buffer[0])));

    uint32_t num_blocks = (in_code_size + runs_per_block - 1) / runs_per_block;
    std::cout << "Decode: code size = " << in_code_size << ", num_blocks = " << num_blocks << ", block size = " << BlockSize << std::endl; 

    for (int i = 0; i < 20; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        // Calculate output offsets using prefix sum of the runs
        // Jefftest 
        GetPrefixSumHipCUB(stream, d_run_buf, d_offset_buf, in_run_buffer.size());

        block_run_length_decode_kernel_offset<BlockSize, RunsPerThread, DecodedItemsPerThread><<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_value_buf, d_offset_buf, d_decoded_buf, in_code_size);
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "<Profiling> hipStreamSynchronize() elapsed time for iteration " << i << ": " << elapsed << " ms" << std::endl;
        float throughput = dec_buf_size * sizeof(uint32_t) / 1000.0 / 1000.0 / 1000.0 * 1000.0 / elapsed;
        std::cout << "<Profiling> RunLengthDecodeHipCUB_Offset throughput based on C++ Chrono lib for iteration " << i << ": " << throughput << " GB/s" << std::endl;
    }

    out_buffer.resize(dec_buf_size);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(out_buffer.data(), d_decoded_buf, dec_buf_size * sizeof(uint32_t), hipMemcpyDeviceToHost));
    *out_buffer_size = dec_buf_size;

    // Jefftest
    /*for (int i = 0; i < dec_buf_size; i++) {
        std::cout << out_buffer[i] << " ";
    }
    std::cout << std::endl;*/
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
        /*if ( i < in_count / 2) {
            in_buffer[i] = i / 2;
        } else {
            in_buffer[i] = i / 5;
        }*/
        in_buffer[i] = i / 10;
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

    // Decode test
    out_value_buffer.resize(code_count);
    out_run_buffer.resize(code_count);
    std::vector<uint32_t> out_buffer_3;
    uint32_t dec_out_size = 0;
    //RunLengthDecodeHipCUB(out_value_buffer, out_run_buffer, out_buffer_3, &dec_out_size);
    RunLengthDecodeHipCUB_OffsetGPU(out_value_buffer, out_run_buffer, out_buffer_3, &dec_out_size);
    std::cout << "Decode buffer size: " << dec_out_size << std::endl;
    CompareVectorsNear(out_buffer_3, in_buffer, 0);
}