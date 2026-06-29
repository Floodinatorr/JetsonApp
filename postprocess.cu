#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "postprocess.cuh"


__global__ void decodeKernel(
    const __half* __restrict__ output,
    int numAnchors, int numAttr,
    float scoreThreshold,
    float scaleX, float scaleY,
    GpuDetection* __restrict__ results,
    int* __restrict__ resultCount,
    int maxResults)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAnchors) return;

    const int N = numAnchors;
    float bestScore = 0.0f;
    int bestClass = -1;

    for (int c = 4; c < numAttr; c++) {
        float cls = __half2float(output[c * N + i]);
        if (cls > bestScore) {
            bestScore = cls;
            bestClass = c - 4;
        }
    }

    if (bestScore < scoreThreshold) return;

    float cx = __half2float(output[0 * N + i]);
    float cy = __half2float(output[1 * N + i]);
    float w  = __half2float(output[2 * N + i]);
    float h  = __half2float(output[3 * N + i]);

    int left   = static_cast<int>((cx - 0.5f * w) * scaleX);
    int top    = static_cast<int>((cy - 0.5f * h) * scaleY);
    int width  = static_cast<int>(w * scaleX);
    int height = static_cast<int>(h * scaleY);

    int idx = atomicAdd(resultCount, 1);
    if (idx < maxResults) {
        results[idx].left = left;
        results[idx].top = top;
        results[idx].width = width;
        results[idx].height = height;
        results[idx].score = bestScore;
        results[idx].classId = bestClass;
    }
}

void launchDecodeKernel(
    const __half* output, int numAnchors, int numAttr,
    float scoreThreshold, float scaleX, float scaleY,
    GpuDetection* results, int* resultCount, int maxResults,
    cudaStream_t stream)
{
    cudaMemsetAsync(resultCount, 0, sizeof(int), stream);

    int threads = 256;
    int blocks = (numAnchors + threads - 1) / threads;
    decodeKernel<<<blocks, threads, 0, stream>>>(
        output, numAnchors, numAttr, scoreThreshold, scaleX, scaleY,
        results, resultCount, maxResults);
}