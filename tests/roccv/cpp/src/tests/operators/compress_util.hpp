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

#include <core/hip_assert.h>
#include <hipcub/device/device_scan.hpp>
#include <hipcub/device/device_scan.hpp>
#include <hipcub/device/device_run_length_encode.hpp>
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_run_length_decode.hpp>
#include <hipcub/block/block_store.hpp>

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

void GetPrefixSumHipCUB(hipStream_t stream, uint32_t *d_in, uint32_t *d_out, int num_items) {
    void* d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    HIP_VALIDATE_NO_ERRORS(hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_temp_storage, temp_storage_bytes));
    HIP_VALIDATE_NO_ERRORS(hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream));
    HIP_VALIDATE_NO_ERRORS(hipFree(d_temp_storage));
}

void RunLengthEncodeHipCUB(hipStream_t stream, uint32_t *d_in, uint32_t* d_value_out, uint32_t* d_run_out, uint32_t* d_runs_count_output, uint32_t in_buf_size) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    HIP_VALIDATE_NO_ERRORS(hipcub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes, d_in, d_value_out, d_run_out, d_runs_count_output, in_buf_size, stream));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_temp_storage, temp_storage_bytes));
    HIP_VALIDATE_NO_ERRORS(hipcub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in, d_value_out, d_run_out, d_runs_count_output, in_buf_size, stream));
    HIP_VALIDATE_NO_ERRORS(hipFree(d_temp_storage));
}

#if 0 // Jefftest
template<uint32_t BlockSize, uint32_t RunsPerThread, uint32_t DecodedItemsPerThread>
__global__
__launch_bounds__(BlockSize)
void block_run_length_decode_kernel(const uint32_t* d_run_items, const uint32_t* d_run_lengths, uint32_t* d_decoded_items) {
    using BlockRunLengthDecodeT = hipcub::BlockRunLengthDecode<uint32_t, BlockSize, RunsPerThread, DecodedItemsPerThread>;
    static constexpr unsigned int decoded_items_per_block = BlockSize * DecodedItemsPerThread;
    __shared__ typename BlockRunLengthDecodeT::TempStorage temp_storage;

    uint32_t run_items[RunsPerThread];
    uint32_t run_lengths[RunsPerThread];

    const uint32_t global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;
    hipcub::LoadDirectBlocked(global_thread_idx, d_run_items, run_items);
    hipcub::LoadDirectBlocked(global_thread_idx, d_run_lengths, run_lengths);

    uint32_t total_decoded_size{};
    BlockRunLengthDecodeT block_run_length_decode(temp_storage, run_items, run_lengths, total_decoded_size);

    uint32_t decoded_window_offset = 0;
    while(decoded_window_offset < total_decoded_size)
    {
        uint32_t decoded_items[DecodedItemsPerThread];
        block_run_length_decode.RunLengthDecode(decoded_items, decoded_window_offset);
        // Jefftest hipcub::StoreDirectBlocked(global_thread_idx, d_decoded_items + decoded_window_offset, decoded_items, hipcub::Min{}(total_decoded_size - decoded_window_offset, decoded_items_per_block));
        hipcub::StoreDirectBlocked(global_thread_idx, d_decoded_items + decoded_window_offset, decoded_items);
        decoded_window_offset += decoded_items_per_block;
    }
}
#endif

#if 0
template<uint32_t BlockSize, uint32_t RunsPerThread, uint32_t DecodedItemsPerThread>
__global__
__launch_bounds__(BlockSize)
void block_run_length_decode_kernel_offset(const uint32_t* d_run_items, const uint32_t* d_run_offsets, uint32_t* d_decoded_items, uint32_t code_size) {
    using BlockRunLengthDecodeT = hipcub::BlockRunLengthDecode<uint32_t, BlockSize, RunsPerThread, DecodedItemsPerThread>;

    uint32_t run_items[RunsPerThread];
    uint32_t run_offsets[RunsPerThread];

    const unsigned global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;

    if (global_thread_idx >= code_size) return;

    hipcub::LoadDirectBlocked(global_thread_idx, d_run_items, run_items);
    hipcub::LoadDirectBlocked(global_thread_idx, d_run_offsets, run_offsets);

    BlockRunLengthDecodeT block_run_length_decode(run_items, run_offsets);

    const uint32_t total_decoded_size = d_run_offsets[(hipBlockIdx_x + 1) * BlockSize * RunsPerThread] - d_run_offsets[hipBlockIdx_x * BlockSize * RunsPerThread];
    
    uint32_t decoded_window_offset = 0;
#pragma unroll
    while (decoded_window_offset < total_decoded_size)
    {
        uint32_t decoded_items[DecodedItemsPerThread];
        block_run_length_decode.RunLengthDecode(decoded_items, decoded_window_offset);
        hipcub::StoreDirectBlocked(global_thread_idx, d_decoded_items + decoded_window_offset, decoded_items);
        decoded_window_offset += BlockSize * DecodedItemsPerThread;
    }
}
/*template<uint32_t BlockSize, uint32_t RunsPerThread, uint32_t DecodedItemsPerThread>
__global__
__launch_bounds__(BlockSize)
void block_run_length_decode_kernel_offset(const uint32_t* d_run_items, const uint32_t* d_run_offsets, uint32_t* d_decoded_items) {
    const unsigned global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;
    // Assuming one thread one run
    const uint32_t run_length = d_run_offsets[global_thread_idx + 1] - d_run_offsets[global_thread_idx];
    const uint32_t offset = d_run_offsets[global_thread_idx];
    const uint32_t value = d_run_items[global_thread_idx];
#pragma unroll
    for (int i = 0; i < run_length; i++) {
        //d_decoded_items[offset + i] = d_run_items[global_thread_idx];
        d_decoded_items[offset + i] = value;
    }
}*/
#endif

#if 1
template<uint32_t BlockSize, uint32_t RunsPerThread, uint32_t DecodedItemsPerThread>
__global__
__launch_bounds__(BlockSize)
void block_run_length_decode_kernel_offset(const uint32_t* d_run_items, const uint32_t* d_run_offsets, uint32_t* d_decoded_items, uint32_t code_size) {
    const unsigned global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;

    if (global_thread_idx >= code_size) return;

    // Assuming one thread per run
    const uint32_t offset = global_thread_idx == 0 ? 0 : d_run_offsets[global_thread_idx - 1];
    const uint32_t run_length = d_run_offsets[global_thread_idx] - offset;
    const uint32_t value = d_run_items[global_thread_idx];
#pragma unroll
    for (int i = 0; i < run_length; i++) {
        d_decoded_items[offset + i] = value;
    }
}
#else
template<uint32_t BlockSize, uint32_t RunsPerThread, uint32_t DecodedItemsPerThread>
__global__
__launch_bounds__(BlockSize)
void block_run_length_decode_kernel_offset(const uint32_t* d_run_items, const uint32_t* d_run_offsets, uint32_t* d_decoded_items, uint32_t code_size) {
    const unsigned global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;

    if (global_thread_idx >= code_size) return;

    __shared__ uint32_t shared_offsets[BlockSize * RunsPerThread];
    const unsigned local_thread_idx = hipThreadIdx_x;
    shared_offsets[local_thread_idx] = d_run_offsets[global_thread_idx];
    __syncthreads();

    // Assuming one thread per run
    const uint32_t offset = global_thread_idx == 0 ? 0 : (local_thread_idx == 0 ? d_run_offsets[global_thread_idx - 1] : shared_offsets[local_thread_idx - 1]);
    const uint32_t run_length = shared_offsets[local_thread_idx] - offset;
    const uint32_t value = d_run_items[global_thread_idx];
#pragma unroll
    for (int i = 0; i < run_length; i++) {
        d_decoded_items[offset + i] = value;
    }
}
#endif

template<uint32_t BlockSize, uint32_t RunsPerThread, uint32_t DecodedItemsPerThread>
void RunLengthDecode(hipStream_t stream, uint32_t num_blocks, uint32_t block_size, uint32_t *d_runs, uint32_t *d_values, uint32_t* d_offsets, uint32_t* d_decoded, uint32_t run_value_counts) {
    // Calculate output offsets using prefix sum of the runs
    GetPrefixSumHipCUB(stream, d_runs, d_offsets, run_value_counts);

    block_run_length_decode_kernel_offset<BlockSize, RunsPerThread, DecodedItemsPerThread><<<dim3(num_blocks), dim3(block_size), 0, stream>>>(d_values, d_offsets, d_decoded, run_value_counts);
}
}  // namespace
