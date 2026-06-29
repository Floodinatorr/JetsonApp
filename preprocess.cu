#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "preprocess.cuh"


__global__ void fusedPreprocessKernel(
    const unsigned char* __restrict__ src,
    int srcWidth, int srcHeight, int srcStep,
    __half* __restrict__ dst,
    int dstWidth, int dstHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;

    const float scaleX = static_cast<float>(srcWidth) / dstWidth;
    const float scaleY = static_cast<float>(srcHeight) / dstHeight;

    float srcXf = (x + 0.5f) * scaleX - 0.5f;
    float srcYf = (y + 0.5f) * scaleY - 0.5f;

    int x0 = static_cast<int>(floorf(srcXf));
    int y0 = static_cast<int>(floorf(srcYf));
    float fx = srcXf - x0;
    float fy = srcYf - y0;

    int x0c = max(0, min(x0, srcWidth - 1));
    int y0c = max(0, min(y0, srcHeight - 1));
    int x1c = max(0, min(x0 + 1, srcWidth - 1));
    int y1c = max(0, min(y0 + 1, srcHeight - 1));

    const unsigned char* p00 = src + y0c * srcStep + x0c * 3;
    const unsigned char* p01 = src + y0c * srcStep + x1c * 3;
    const unsigned char* p10 = src + y1c * srcStep + x0c * 3;
    const unsigned char* p11 = src + y1c * srcStep + x1c * 3;

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;


    float b = p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11;
    float g = p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11;
    float r = p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11;


    const float invScale = 1.0f / 255.0f;
    r *= invScale;
    g *= invScale;
    b *= invScale;

    const int hw = dstWidth * dstHeight;
    const int idx = y * dstWidth + x;

    dst[idx]          = __float2half(r);
    dst[hw + idx]      = __float2half(g);
    dst[2 * hw + idx] = __float2half(b);
}

void launchFusedPreprocess(
    const unsigned char* src, int srcWidth, int srcHeight, int srcStep,
    __half* dst, int dstWidth, int dstHeight,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    fusedPreprocessKernel<<<grid, block, 0, stream>>>(src, srcWidth, srcHeight, srcStep,
                                                       dst, dstWidth, dstHeight);
}