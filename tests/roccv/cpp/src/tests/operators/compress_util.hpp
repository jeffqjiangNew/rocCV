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
#include <hipcub/device/device_reduce.hpp>
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

void RunLengthEncodeCPU(uint32_t *input, uint32_t input_size, uint32_t *output_values, uint32_t *output_runs, uint32_t* output_size) {
    int pair_index = 0;
    int value = input[0];
    int run_length = 1;

    for (int i = 1; i < input_size; i++) {
        if (input[i] != value) {
            output_values[pair_index] = value;
            output_runs[pair_index] = run_length;
            pair_index++;

            value = input[i];
            run_length = 1;
        } else {
            run_length++;
        }
    }

    // Last pair
    output_values[pair_index] = value;
    output_runs[pair_index] = run_length;
    pair_index++;

    *output_size = pair_index;
    // Jefftest
    std::cout << "RLE value array: ";
    for (int i = 0; i < *output_size; i++) {
        std::cout << output_values[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "RLE run array: ";
    for (int i = 0; i < *output_size; i++) {
        std::cout << output_runs[i] << ", ";
    }
    std::cout << std::endl;
}

void RunLengthDecodeCPU(uint32_t *input_values, uint32_t *input_runs, uint32_t code_size, uint32_t *output, uint32_t* output_size) {
    uint32_t output_index = 0;

    for (uint32_t i = 0; i < code_size; i++) {
        for (uint32_t j = 0; j < input_runs[i]; j++) {
            output[output_index] = input_values[i];
            output_index++;
        }
    }

    *output_size = output_index;
}


void DeltaEncodeCPU(uint32_t *input, uint32_t input_size, uint32_t *output) {
    uint32_t prev_value = input[0];
    output[0] = input[0];
    for (int i = 1; i < input_size; i++) {
        output[i] = input[i] - prev_value;
        prev_value = input[i];
    }
    // Jefftest
    std::cout << "Delta encoded output array: ";
    for (int i = 0; i < input_size; i++) {
        // std::cout << output[i] << " ";
        std::cout << static_cast<int32_t>(output[i]) << " ";
    }
    std::cout << std::endl;
}

void DeltaDecodeCPU(uint32_t *input, uint32_t input_size, uint32_t *output) {
    uint32_t prev_value = input[0];
    output[0] = input[0];
    for (int i = 1; i < input_size; i++) {
        output[i] = prev_value + input[i];
        prev_value = output[i];
    }

    // Jefftest
    /*std::cout << "Delta decoded output array: ";
    for (int i = 0; i < input_size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;*/
}

// Bit packing: fixed B bits per value, LSB-first into byte stream.
inline uint32_t BitPackOutputBytes(uint32_t num_items, int bits_per_value) {
    return (num_items * static_cast<uint32_t>(bits_per_value) + 7u) / 8u;
}

void BitPackCPU(uint32_t* input, uint32_t input_size, uint8_t* output, uint32_t* output_byte_size, int bits_per_value = 8) {
    const uint32_t mask = (1u << bits_per_value) - 1u;
    uint32_t bit_cursor = 0;
    for (uint32_t i = 0; i < input_size; i++) {
        uint32_t v = input[i] & mask;
        for (int b = 0; b < bits_per_value; b++) {
            uint32_t byte_idx = bit_cursor / 8;
            int bit_idx = bit_cursor % 8;
            if (v & (1u << b))
                output[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
            bit_cursor++;
        }
    }
    *output_byte_size = BitPackOutputBytes(input_size, bits_per_value);
}

void BitUnpackCPU(uint8_t* input, uint32_t input_byte_size, uint32_t* output, uint32_t output_size, int bits_per_value = 8) {
    uint32_t bit_cursor = 0;
    const uint32_t total_bits = output_size * static_cast<uint32_t>(bits_per_value);
    for (uint32_t i = 0; i < output_size; i++) {
        uint32_t v = 0;
        for (int b = 0; b < bits_per_value; b++) {
            if (bit_cursor >= total_bits) break;
            uint32_t byte_idx = bit_cursor / 8;
            int bit_idx = bit_cursor % 8;
            if (byte_idx < input_byte_size && (input[byte_idx] & (1u << bit_idx)))
                v |= 1u << b;
            bit_cursor++;
        }
        output[i] = v;
    }
}

// Minimum number of bits needed to represent max_val (1..32). Use after finding max of array.
inline int BitsForMax(uint32_t max_val) {
    if (max_val == 0u) return 1;
    return 32 - __builtin_clz(max_val);
}

// Host: scan array once, return bits_per_value so all values fit.
inline int BitsPerValueFromArrayCPU(const uint32_t* input, uint32_t n) {
    if (n == 0u) return 1;
    uint32_t max_val = input[0];
    for (uint32_t i = 1; i < n; i++)
        if (input[i] > max_val) max_val = input[i];
    return BitsForMax(max_val);
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

void MaxValueDevice(hipStream_t stream, const uint32_t* d_in, uint32_t num_items, uint32_t* d_max) {
    if (num_items == 0u) {
        HIP_VALIDATE_NO_ERRORS(hipMemset(d_max, 0, sizeof(uint32_t)));
        return;
    }
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    HIP_VALIDATE_NO_ERRORS(hipcub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, static_cast<int>(num_items), stream));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_temp_storage, temp_storage_bytes));
    HIP_VALIDATE_NO_ERRORS(hipcub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, static_cast<int>(num_items), stream));
    HIP_VALIDATE_NO_ERRORS(hipFree(d_temp_storage));
}

int BitsPerValueFromArrayDevice(hipStream_t stream, const uint32_t* d_in, uint32_t num_items) {
    if (num_items == 0u) return 1;
    uint32_t* d_max = nullptr;
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d_max, sizeof(uint32_t)));
    MaxValueDevice(stream, d_in, num_items, d_max);
    uint32_t max_val = 0u;
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(&max_val, d_max, sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_VALIDATE_NO_ERRORS(hipFree(d_max));
    return BitsForMax(max_val);
}

__global__ void bit_pack_kernel(const uint32_t* __restrict__ d_in, uint8_t* __restrict__ d_out, uint32_t num_items, int bits_per_value) {
    const uint32_t total_bytes = (num_items * bits_per_value + 7) / 8;
    for (uint32_t k = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; k < total_bytes; k += hipGridDim_x * hipBlockDim_x) {
        uint8_t byte_val = 0;
        const uint32_t first_bit = k * 8u;
        const uint32_t last_bit = first_bit + 7u;
        for (uint32_t s = first_bit; s <= last_bit && s < num_items * static_cast<uint32_t>(bits_per_value); s++) {
            uint32_t e = s / bits_per_value;
            int bit_in_elem = s - e * bits_per_value;
            if ((d_in[e] >> bit_in_elem) & 1u)
                byte_val |= static_cast<uint8_t>(1u << (s - first_bit));
        }
        d_out[k] = byte_val;
    }
}

__global__ void bit_unpack_kernel(const uint8_t* __restrict__ d_in, uint32_t* __restrict__ d_out, uint32_t num_items, int bits_per_value) {
    for (uint32_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < num_items; i += hipGridDim_x * hipBlockDim_x) {
        uint32_t v = 0;
        const uint32_t start_bit = i * bits_per_value;
        for (int b = 0; b < bits_per_value; b++) {
            uint32_t s = start_bit + b;
            uint32_t byte_idx = s / 8;
            int bit_idx = s % 8;
            if ((d_in[byte_idx] >> bit_idx) & 1u)
                v |= 1u << b;
        }
        d_out[i] = v;
    }
}

void BitPackDevice(hipStream_t stream, const uint32_t* d_in, uint32_t num_items, uint8_t* d_out, int bits_per_value = 8) {
    if (num_items == 0) return;
    const uint32_t total_bytes = BitPackOutputBytes(num_items, bits_per_value);
    const uint32_t block_size = 256u;
    const uint32_t num_blocks = (total_bytes + block_size - 1u) / block_size;
    bit_pack_kernel<<<num_blocks, block_size, 0, stream>>>(d_in, d_out, num_items, bits_per_value);
}

void BitUnpackDevice(hipStream_t stream, const uint8_t* d_in, uint32_t packed_byte_size, uint32_t* d_out, uint32_t num_items, int bits_per_value = 8) {
    if (num_items == 0) return;
    const uint32_t block_size = 256u;
    const uint32_t num_blocks = (num_items + block_size - 1u) / block_size;
    bit_unpack_kernel<<<num_blocks, block_size, 0, stream>>>(d_in, d_out, num_items, bits_per_value);
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
