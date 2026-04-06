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

#include <fstream>
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

void TestRLE() {
    printf("Running RLE compress/decompress test ...\n");
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

void CascadeEncode(std::vector<uint32_t>& in_buffer, std::vector<uint8_t>& value_buffer, uint32_t *value_buffer_size, std::vector<uint8_t>& run_buffer, uint32_t *run_buffer_size) {
    uint32_t in_buf_size = in_buffer.size();
    uint32_t max_out_buf_size = in_buf_size * sizeof(in_buffer[0]); // out buffer type is uint8_t
    value_buffer.resize(max_out_buf_size);
    run_buffer.resize(max_out_buf_size);

    // RLE encode
    std::vector<uint32_t> rle_value_buffer(in_buf_size);
    std::vector<uint32_t> rle_run_buffer(in_buf_size);
    uint32_t rle_code_size = 0;
    RunLengthEncodeCPU(in_buffer.data(), in_buf_size, reinterpret_cast<uint32_t*>(rle_value_buffer.data()), reinterpret_cast<uint32_t*>(rle_run_buffer.data()), &rle_code_size);

    // Delta encode
    std::vector<uint32_t> delta_value_buffer(rle_code_size);
    std::vector<uint32_t> delta_run_buffer(rle_code_size);
    DeltaEncodeCPU(reinterpret_cast<uint32_t*>(rle_value_buffer.data()), rle_code_size, reinterpret_cast<uint32_t*>(delta_value_buffer.data()));
    DeltaEncodeCPU(reinterpret_cast<uint32_t*>(rle_run_buffer.data()), rle_code_size, reinterpret_cast<uint32_t*>(delta_run_buffer.data()));

    // Bit pack encode
    int bits = BitsPerValueFromArrayCPU(in_buffer.data(), in_buf_size);
    //BitPackCPU(in_buffer.data(), in_buf_size, reinterpret_cast<uint8_t*>(value_buffer.data()), reinterpret_cast<uint32_t*>(value_buffer_size), bits);
    //BitPackCPU(in_buffer.data(), in_buf_size, reinterpret_cast<uint8_t*>(run_buffer.data()), reinterpret_cast<uint32_t*>(run_buffer_size), bits);
    
    // Copy the delta_value_buffer contents into value_buffer (uint8_t format)
    std::memcpy(value_buffer.data(), delta_value_buffer.data(), rle_code_size * sizeof(uint32_t));
    // Copy the delta_run_buffer contents into run_buffer (uint8_t format)
    std::memcpy(run_buffer.data(), delta_run_buffer.data(), rle_code_size * sizeof(uint32_t));
    *value_buffer_size = rle_code_size * sizeof(in_buffer[0]);
    *run_buffer_size = rle_code_size * sizeof(in_buffer[0]);

    value_buffer.resize(*value_buffer_size);
    run_buffer.resize(*run_buffer_size);
}

void CascadeDecode(std::vector<uint8_t>& value_buffer, uint32_t value_buffer_size, std::vector<uint8_t>& run_buffer, uint32_t run_buffer_size, std::vector<uint32_t>& dec_buffer, uint32_t *dec_buf_size) {

    // Delta decode
    uint32_t code_size = value_buffer_size / sizeof(uint32_t);
    std::vector<uint32_t> rle_value_buffer(code_size);
    std::vector<uint32_t> rle_run_buffer(code_size);
    DeltaDecodeCPU(reinterpret_cast<uint32_t*>(value_buffer.data()), code_size, reinterpret_cast<uint32_t*>(rle_value_buffer.data()));
    DeltaDecodeCPU(reinterpret_cast<uint32_t*>(run_buffer.data()), code_size, reinterpret_cast<uint32_t*>(rle_run_buffer.data()));

    // RLE decode
    RunLengthDecodeCPU(reinterpret_cast<uint32_t*>(rle_value_buffer.data()), reinterpret_cast<uint32_t*>(rle_run_buffer.data()), code_size, reinterpret_cast<uint32_t*>(dec_buffer.data()), dec_buf_size); 
    
}

int main(int argc, char **argv) {

    std::string input_filename;
    std::vector<uint32_t> input_data;

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            if ((std::string(argv[i]) == "-i" || std::string(argv[i]) == "--input") && i + 1 < argc) {
                input_filename = argv[i + 1];
                ++i; // Skip filename
            }
            else if ((std::string(argv[i]) == "-t" || std::string(argv[i]) == "--type") && i + 1 < argc) {
                std::string data_type = argv[i + 1];
                std::cout << "Data type specified: " << data_type << std::endl;
                ++i; // Skip type argument
            }
        }
    } else {
        std::cout << "No input file specified. Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    if (!input_filename.empty()) {
        std::cout << "Input file provided: " << input_filename << std::endl;
        std::ifstream infile(input_filename, std::ios::binary);
        if (!infile.is_open()) {
            std::cerr << "Error: Could not open input file '" << input_filename << "'." << std::endl;
            return 1;
        }
        // For an ASCII file, where each line is a decimal integer, read line by line and fill a vector<uint32_t>
        std::string line;
        while (std::getline(infile, line)) {
            if (!line.empty()) {
                //std::cout << "Read line: " << line << std::endl; // Jefftest
                try {
                    uint32_t value = static_cast<uint32_t>(std::stoul(line));
                    //std::cout << "Read value: " << value << std::endl; // Jefftest
                    input_data.push_back(value);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Failed to parse line as integer: '" << line << "'" << std::endl;
                } catch (const std::out_of_range& e) {
                    //std::cerr << "Warning: Value out of uint32_t range: '" << line << "'" << std::endl;
                    std::cout << "Warning: Value out of uint32_t range: '" << line << "'" << std::endl;
                }
            }
        }
        infile.close();

        std::cout << "Input data size: " << input_data.size() << " elements" << std::endl;
    } else {
        std::cout << "No input file (-i <filename>) specified." << std::endl;
    }

    TestRLE();

    #if 0
    uint32_t in_count = 1024;
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
    #endif
    uint32_t in_count = input_data.size();
    std::vector<uint8_t> enc_value_buffer;
    uint32_t enc_value_buf_size;
    std::vector<uint8_t> enc_run_buffer;
    uint32_t enc_run_buf_size;
    CascadeEncode(input_data, enc_value_buffer, &enc_value_buf_size, enc_run_buffer, &enc_run_buf_size);
    std::cout << "Compression ratio: " << static_cast<float>(in_count * sizeof(input_data[0])) / static_cast<float>(enc_value_buf_size + enc_run_buf_size)  << std::endl;

    std::vector<uint32_t> dec_buffer(in_count);
    uint32_t dec_buf_size = 0;
    CascadeDecode(enc_value_buffer, enc_value_buf_size, enc_run_buffer, enc_run_buf_size, dec_buffer, &dec_buf_size);
    CompareVectorsNear(dec_buffer, input_data, 0);

    return 0;
}