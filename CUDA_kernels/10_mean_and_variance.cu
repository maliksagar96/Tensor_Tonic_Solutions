#include <cuda_runtime.h>

__global__ void meanKernel(const float* input, float* mean_out, int N) {

  int tx = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + tx;

  extern __shared__ float shmem[];

  if(tid < N) {
    shmem[tx] = input[tid];
  }
  else {
    shmem[tx] = 0.0f;
  }

  __syncthreads();

  for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if(tx < stride) {
      shmem[tx] += shmem[tx + stride];
    }
    __syncthreads();
  }

  if(tx == 0) {
    atomicAdd(mean_out, shmem[0]);
  }
}

__global__ void varianceKernel(const float* input, const float* mean_out, float* var_out, int N) {

  int tx = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + tx;

  extern __shared__ float shmem[];

  float mean = *mean_out;

  if(tid < N) {
    float diff = input[tid] - mean;
    shmem[tx] = diff * diff;
  }
  else {
    shmem[tx] = 0.0f;
  }

  __syncthreads();

  for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if(tx < stride) {
      shmem[tx] += shmem[tx + stride];
    }
    __syncthreads();
  }

  if(tx == 0) {
    atomicAdd(var_out, shmem[0]);
  }
}

__global__ void divideKernel(float* value, int N) {
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    *value /= (float)N;
  }
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  cudaMemset(mean_out, 0, sizeof(float));
  cudaMemset(var_out, 0, sizeof(float));

  meanKernel<<<blocks, threads, threads * sizeof(float)>>>(input, mean_out, N);

  divideKernel<<<1,1>>>(mean_out, N);

  varianceKernel<<<blocks, threads, threads * sizeof(float)>>>(input, mean_out, var_out, N);

  divideKernel<<<1,1>>>(var_out, N);

  cudaDeviceSynchronize();
}