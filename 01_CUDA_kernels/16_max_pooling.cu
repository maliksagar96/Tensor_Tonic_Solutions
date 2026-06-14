#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_2d_kernel(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int outH = (H - kH) / sH + 1;
  int outW = (W - kW) / sW + 1;

  if(row < outH && col < outW) {

    int inRow = row * sH;
    int inCol = col * sW;

    float maxVal = -FLT_MAX;

    for(int ky = 0; ky < kH; ++ky) {
      for(int kx = 0; kx < kW; ++kx) {
        maxVal = max(maxVal,input[(inRow + ky) * W + (inCol + kx)]);
      }
    }

    output[row * outW + col] = maxVal;
  }
}

extern "C" void solve(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
	int outH = (H - kH) / sH + 1;
	int outW = (W - kW) / sW + 1;
	dim3 threads(16, 16);
	dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
	max_pool_2d_kernel<<<blocks, threads>>>(input, output, H, W, kH, kW, sH, sW);
	cudaDeviceSynchronize();
}
