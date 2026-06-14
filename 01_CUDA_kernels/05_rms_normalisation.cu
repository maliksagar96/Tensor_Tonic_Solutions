#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < M) {

    float mean_square = 0.0f;

    for(int col = 0; col < N; col++) {
      float value = input[row * N + col];
      mean_square += value * value;
    }

    mean_square /= N;

    float rms_inv = rsqrtf(mean_square + eps);

    for(int col = 0; col < N; col++) {
      output[row * N + col] =
        input[row * N + col] * rms_inv * gamma[col];
    }
  }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {

  int threads = 256;
  int blocks = (M + threads - 1) / threads;

  rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);

  cudaDeviceSynchronize();
}