#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct GpuDetection {
    int left;
    int top;
    int width;
    int height;
    float score;
    int classId;
};

void launchDecodeKernel(
    const __half* output, int numAnchors, int numAttr,
    float scoreThreshold, float scaleX, float scaleY,
    GpuDetection* results, int* resultCount, int maxResults,
    cudaStream_t stream);