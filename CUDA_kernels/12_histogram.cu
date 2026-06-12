/*
  This is not convoltion it is cross correction. 
*/
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int outN, int kN) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < outN) {

    float sum = 0.0f;

    for(int k = 0; k < kN; ++k) {
      sum += input[tid + k] * kernel[k];
    }

    output[tid] = sum;
  }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {

  int outN = N - kN + 1;

  int threads = 256;
  int blocks = (outN + threads - 1) / threads;

  conv1d_kernel<<<blocks, threads>>>(input, kernel, output, outN, kN);

  cudaDeviceSynchronize();
}