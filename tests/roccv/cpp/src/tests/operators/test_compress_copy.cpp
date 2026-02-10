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

using namespace roccv::tests;

int main(int argc, char **argv) {
    int num_devices = 0;

    std::cout << "Running compress and copy test ..." << std::endl;
    HIP_VALIDATE_NO_ERRORS(hipGetDeviceCount(&num_devices));
    if (num_devices < 2) {
        std::cout << "Only one GPU found. Test will not run." << std::endl;
        exit(1);
    }

    // Create HIP streams on both devices
    int device_0_id = 0;
    int device_1_id = 1;
    hipStream_t hip_stream_d0;
    hipStream_t hip_stream_d1;
    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_0_id));
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&hip_stream_d0));
    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_1_id));
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&hip_stream_d1));

    // Create a vector and fill it with random data.
    uint32_t num_symbols = 1024 * 1024 * 100;
    std::vector<uint32_t> src_buffer(num_symbols);
    //FillVector(src_buffer);
    for (int i = 0; i < num_symbols; i++) {
        src_buffer[i] = i / 2;
    }
    std::cout << "Raw data symbols count: " << num_symbols << std::endl;

    // Pre-allocate buffers on devices
    uint32_t* d0_src_buf;
    uint32_t* d0_values;
    uint32_t* d0_runs;
    uint32_t* d0_runs_count;
    uint32_t* d1_values;
    uint32_t* d1_runs;
    uint32_t* d1_offsets;
    uint32_t* d1_decoded;

    uint32_t raw_data_size = num_symbols * sizeof(uint32_t);
    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_0_id));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d0_src_buf, raw_data_size));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d0_values, raw_data_size));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d0_runs, raw_data_size));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d0_runs_count, sizeof(uint32_t)));

    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_1_id));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d1_values, raw_data_size));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d1_runs, raw_data_size));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d1_offsets, raw_data_size));
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&d1_decoded, raw_data_size));

    // Work on deviec 0
    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_0_id));
    int can_access_peer = 0;
    HIP_VALIDATE_NO_ERRORS(hipDeviceCanAccessPeer(&can_access_peer, device_0_id, device_1_id));
    std::cout << "Device " << device_0_id << " can access peer device " << device_1_id << ": " << can_access_peer << " .................." << std::endl;
    if (can_access_peer == 0) {
        std::cout << "Peer device access not enabled. Test will not run." << std::endl;
        exit(1);
    }
    // Enable direct access to memory allocations on device 1
    HIP_VALIDATE_NO_ERRORS(hipDeviceEnablePeerAccess(device_1_id, 0));

    // Copy data to device 0
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(d0_src_buf, src_buffer.data(), raw_data_size, hipMemcpyHostToDevice));
    
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();

	// Work on device 0
    	HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_0_id));

        // Compress on device 0
        RunLengthEncodeHipCUB(hip_stream_d0, d0_src_buf, d0_values, d0_runs, d0_runs_count, num_symbols);

        // Copy compressed data from device 0 to device 1
        uint32_t code_count;
        HIP_VALIDATE_NO_ERRORS(hipMemcpyWithStream(&code_count, d0_runs_count, sizeof(uint32_t), hipMemcpyDeviceToHost, hip_stream_d0));
        HIP_VALIDATE_NO_ERRORS(hipMemcpyPeerAsync(d1_values, device_1_id, d0_values, device_0_id, code_count * sizeof(uint32_t), hip_stream_d0));
        HIP_VALIDATE_NO_ERRORS(hipMemcpyPeerAsync(d1_runs, device_1_id, d0_runs, device_0_id, code_count * sizeof(uint32_t), hip_stream_d0));
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(hip_stream_d0));

        // Work on device 1
        HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_1_id));

        // De-compress on device 1
        const uint32_t BlockSize = 32; // 256; // Jefftest 64;
        const uint32_t RunsPerThread = 1; // 2;
        const uint32_t DecodedItemsPerThread = 1; // 2;
        constexpr auto runs_per_block  = BlockSize * RunsPerThread;
        uint32_t num_blocks = (code_count + runs_per_block - 1) / runs_per_block;
        RunLengthDecode<BlockSize, RunsPerThread, DecodedItemsPerThread>(hip_stream_d1, num_blocks, BlockSize, d1_runs, d1_values, d1_offsets, d1_decoded, code_count);
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(hip_stream_d1));

        std::cout << "Raw symbol count: " << num_symbols << ", RLE code size = " << code_count * 2 << std::endl;
        std::cout << "RLE compression ratio: " << static_cast<float>(num_symbols) / (static_cast<float>(code_count) * 2.0f)  << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "<Profiling> Compress-copy-decompress elapsed time for iteration " << i << ": " << elapsed << " ms" << std::endl;
        float throughput = raw_data_size / 1000.0 / 1000.0 / 1000.0 * 1000.0 / elapsed;
        std::cout << "<Profiling> Compress-copy-decompress throughput based on C++ Chrono lib for iteration " << i << ": " << throughput << " GB/s" << std::endl;
    }

    // Copy de-compressed data to Host and compare to the original
    std::vector<uint32_t> dec_buffer(num_symbols);
    HIP_VALIDATE_NO_ERRORS(hipMemcpyWithStream(dec_buffer.data(), d1_decoded, raw_data_size, hipMemcpyDeviceToHost, hip_stream_d1));
    CompareVectorsNear(dec_buffer, src_buffer, 0);
    std::cout << "Compress and copy test completed without errors." << std::endl;

    // Measure raw data copy throughput
    // Work on deviec 0
    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_0_id));
    // Enable direct access to memory allocations on device 1
    //HIP_VALIDATE_NO_ERRORS(hipDeviceEnablePeerAccess(device_1_id, 0));

    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        HIP_VALIDATE_NO_ERRORS(hipMemcpyPeerAsync(d1_decoded, device_1_id, d0_src_buf, device_0_id, raw_data_size, hip_stream_d0));
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(hip_stream_d0));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "<Profiling> Direct copy (no compression) elapsed time for iteration " << i << ": " << elapsed << " ms" << std::endl;
        float throughput = raw_data_size / 1000.0 / 1000.0 / 1000.0 * 1000.0 / elapsed;
        std::cout << "<Profiling> Direct copy (no compression) throughput based on C++ Chrono lib for iteration " << i << ": " << throughput << " GB/s" << std::endl;
    }

    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_0_id));
    HIP_VALIDATE_NO_ERRORS(hipFree(d0_src_buf));
    HIP_VALIDATE_NO_ERRORS(hipFree(d0_values));
    HIP_VALIDATE_NO_ERRORS(hipFree(d0_runs));
    HIP_VALIDATE_NO_ERRORS(hipFree(d0_runs_count));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(hip_stream_d0));
    HIP_VALIDATE_NO_ERRORS(hipSetDevice(device_1_id));
    HIP_VALIDATE_NO_ERRORS(hipFree(d1_values));
    HIP_VALIDATE_NO_ERRORS(hipFree(d1_runs));
    HIP_VALIDATE_NO_ERRORS(hipFree(d1_offsets));
    HIP_VALIDATE_NO_ERRORS(hipFree(d1_decoded));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(hip_stream_d1));
}
