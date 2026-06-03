#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

__device__ float d_sum;
__device__ float d_max;

__global__ void findMax(float *nums, float *reduced_nums, int N) {

	extern __shared__ float shmem[];
  int tx = threadIdx.x;
  int tid1 = 2 * blockDim.x * blockIdx.x + tx;
  int tid2 = tid1 + blockDim.x;

  float blockMax = -FLT_MAX;

  if(tid1 < N)
    blockMax = fmaxf(blockMax, nums[tid1]);

  if(tid2 < N)
    blockMax = fmaxf(blockMax,nums[tid2]);

  shmem[tx] = blockMax;

  __syncthreads();

  for(int stride = blockDim.x / 2; stride > 32; stride /= 2) {

    if(tx < stride)
      shmem[tx] = fmaxf(shmem[tx], shmem[tx + stride]);

    __syncthreads();
  }

  if(tx < 32) {
    blockMax = fmaxf(shmem[tx], shmem[tx + 32]);		
    unsigned mask = 0xffffffff;
		//No need to write syncthreads() in between because the threads in a warp are already in sync. 
    blockMax = fmaxf(blockMax, __shfl_down_sync(mask, blockMax, 16));
    blockMax = fmaxf(blockMax, __shfl_down_sync(mask, blockMax, 8));
    blockMax = fmaxf(blockMax, __shfl_down_sync(mask, blockMax, 4));
    blockMax = fmaxf(blockMax, __shfl_down_sync(mask, blockMax, 2));
    blockMax = fmaxf(blockMax, __shfl_down_sync(mask, blockMax, 1));

    if(tx == 0)
      reduced_nums[blockIdx.x] = blockMax;

			if(gridDim.x == 1) {
				d_max = blockMax;
			}
  }

}

__global__ void findSum(float *nums, float *reduced_nums, int N) {

	extern __shared__ float shmem[];
  int tx = threadIdx.x;
  int tid1 = 2 * blockDim.x * blockIdx.x + tx;
  int tid2 = tid1 + blockDim.x;

  float sum = 0;

  if(tid1 < N)
    sum += nums[tid1];

  if(tid2 < N)
    sum += nums[tid2];

  shmem[tx] = sum;

  __syncthreads();

  for(int stride = blockDim.x / 2; stride > 32; stride /= 2) {

    if(tx < stride)
      shmem[tx] += shmem[tx + stride];

    __syncthreads();
  }

  if(tx < 32) {
    sum = shmem[tx] + shmem[tx + 32];		
    unsigned mask = 0xffffffff;
		//No need to write syncthreads() in between because the threads in a warp are already in sync. 
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);

    if(tx == 0)
      reduced_nums[blockIdx.x] = sum;

			if(gridDim.x == 1) {
				d_sum = sum;
			}
  }

}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write code here
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < N) {
		float x = exp(input[tid] - d_max);
		output[tid] = x/d_sum;
	}
}

__global__ void computeExp(const float *input, float *d_exp, int N) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < N) {
		d_exp[tid] = expf(input[tid] - d_max);
	}
}

extern "C" void solve(const float* input, float* output, int N) {

  int threads = 256;
  int blocks = (N + 2 * threads - 1)/(2 * threads);

  float *d_reduced;
  float *d_exp;

  cudaMalloc(&d_reduced, blocks * sizeof(float));
  cudaMalloc(&d_exp, N * sizeof(float));

  findMax<<<blocks, threads, threads * sizeof(float)>>>((float*)input, d_reduced, N);

  int currN = blocks;

  while(currN > 1) {

    int nextBlocks = (currN + 2 * threads - 1)/(2 * threads);

    findMax<<<nextBlocks, threads, threads * sizeof(float)>>>(d_reduced, d_reduced, currN);

    currN = nextBlocks;
  }

  computeExp<<<(N + threads - 1)/threads, threads>>>((float*)input, d_exp, N);

  blocks = (N + 2 * threads - 1)/(2 * threads);

  findSum<<<blocks, threads, threads * sizeof(float)>>>(d_exp, d_reduced, N);

  currN = blocks;

  while(currN > 1) {

    int nextBlocks = (currN + 2 * threads - 1)/(2 * threads);

    findSum<<<nextBlocks, threads, threads * sizeof(float)>>>(d_reduced, d_reduced, currN);

    currN = nextBlocks;
  }

  softmax_kernel<<<(N + threads - 1)/threads, threads>>>(input, output, N);

  cudaDeviceSynchronize();

  cudaFree(d_reduced);
  cudaFree(d_exp);
}