#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  extern __shared__ float shmem[];

  if(tid < N) {
    shmem[tx] = input[tid] * input[tid];
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
    atomicAdd(sumv, shmem[0]);
  }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < N) {
    output[tid] = input[tid] / sqrtf(*sumv);
  }
}

extern "C" void solve(const float* input, float* output, int N) {

  float* d_sum;

  cudaMalloc(&d_sum, sizeof(float));
  cudaMemset(d_sum, 0, sizeof(float));

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  reduce_sq_sum<<<blocks, threads, threads * sizeof(float)>>>(input, d_sum, N);

  divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

  cudaDeviceSynchronize();

  cudaFree(d_sum);
}