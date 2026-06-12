#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output, int D, int H, int W, int kD, int kH, int kW) {

  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int row    = blockIdx.y * blockDim.y + threadIdx.y;
  int depth = blockIdx.z * blockDim.z + threadIdx.z;

  int outH = H - kH + 1;
  int outW = W - kW + 1;
  int outD = D - kD + 1;

  if(row < outH && column < outW && depth < outD) {
    float sum = 0.0f;
    for(int kz = 0;kz < kD;kz++) {
      for(int ky = 0; ky < kH; ++ky) {
        for(int kx = 0; kx < kW; ++kx) {
          sum += input[((depth + kz) * H + (row + ky)) * W + (column + kx)] * kernel[((kz * kH + ky) * kW + kx)];
        }
      }
    }

    output[(depth * outH + row) * outW + column] = sum;
  }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int D, int H, int W, int kD, int kH, int kW) {
  int outD = D - kD + 1;
  int outH = H - kH + 1;
  int outW = W - kW + 1;
  dim3 threads(8, 8, 8);
  dim3 blocks((outW + 7) / 8, (outH + 7) / 8, (outD + 7) / 8);
  conv3d_kernel<<<blocks, threads>>>(input, kernel, output, D, H, W, kD, kH, kW);
  cudaDeviceSynchronize();
}
