#include <cuda_fp16.h>
void launchHwc2Chw(const unsigned char* __restrict__ input, __half* __restrict__ output, int width, int height);
