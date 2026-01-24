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

int main(int argc, char** argv) {
    hipStream_t hip_stream;
    int num_items = 10;
    void* d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    int* d_in = nullptr;
    int* d_out = nullptr;
    int *h_in, *h_out;

    h_in = (int*)malloc(num_items * sizeof(int));
    for (int i = 0; i < num_items; i++) {
        h_in[i] = 1;
    }
    h_out = (int*)malloc(num_items * sizeof(int));
    for (int i = 0; i < num_items; i++) {
        h_out[i] = 0;
    }
    HIP_API_CALL(hipMalloc(&d_in, num_items * sizeof(int)));
    HIP_API_CALL(hipMalloc(&d_out, num_items * sizeof(int)));

    HIP_API_CALL(hipStreamCreate(&hip_stream));
    
    HIP_API_CALL(hipMemcpyAsync(d_in, h_in, num_items * sizeof(int), hipMemcpyHostToDevice, hip_stream));

    HIP_API_CALL(hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, hip_stream));
    HIP_API_CALL(hipMalloc(&d_temp_storage, temp_storage_bytes));
    HIP_API_CALL(hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, hip_stream));

    HIP_API_CALL(hipMemcpyAsync(h_out, d_out, num_items * sizeof(int), hipMemcpyDeviceToHost, hip_stream));
    HIP_API_CALL(hipStreamSynchronize(hip_stream));

    for (int i = 0; i < num_items; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    free(h_in);
    free(h_out);
    HIP_API_CALL(hipFree(d_in));
    HIP_API_CALL(hipFree(d_out));

}

