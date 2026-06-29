#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launchFusedPreprocess(
    const unsigned char* src, int srcWidth, int srcHeight, int srcStep,
    __half* dst, int dstWidth, int dstHeight,
    cudaStream_t stream);