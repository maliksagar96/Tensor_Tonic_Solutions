#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {

  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int row    = blockIdx.y * blockDim.y + threadIdx.y;

  int outH = H - kH + 1;
  int outW = W - kW + 1;

  if(row < outH && column < outW) {
    float sum = 0.0f;
    for(int ky = 0; ky < kH; ++ky) {
      for(int kx = 0; kx < kW; ++kx) {
        sum += input[(row + ky) * W + (column + kx)] * kernel[ky * kW + kx];
      }
    }

    output[row * outW + column] = sum;
  }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
	int outH = H - kH + 1;
	int outW = W - kW + 1;
	dim3 threads(16, 16);
	dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
	conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);
	cudaDeviceSynchronize();
}
