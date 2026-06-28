#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <bits/fs_fwd.h>
#include "preprocess.cuh"

__global__ void hwc2chw(const unsigned char* __restrict__ input, __half* __restrict__ output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int hw = height*width;
    int idx = y*width + x;

    float r = input[idx] * (1.0f / 255.0f);
    float g = input[idx + 1] * (1.0f / 255.0f);
    float b = input[idx + 2] * (1.0f / 255.0f);

    int idxR = idx;
    int idxG = hw + idx;
    int idxB = 2*hw + idx;

    output[idxR] = __float2half(r);
    output[idxG] = __float2half(g);
    output[idxB] = __float2half(b);
};

void launchHwc2Chw(const unsigned char* __restrict__ input, __half* __restrict__ output, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    hwc2chw<<<grid, block>>>(input, output, width, height);
}