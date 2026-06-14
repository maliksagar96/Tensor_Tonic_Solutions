#include <cuda_runtime.h>
#include <float.h>

__device__ float d_max;

__global__ void gpuReduction(float *nums, float *reduced_nums, int N) {

  extern __shared__ float shmem[];
  int tx = threadIdx.x;
  int tid1 = 2 * blockDim.x * blockIdx.x + tx;
  int tid2 = tid1 + blockDim.x;

  float blockMax = -INFINITY;

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

__global__ void max_kernel(const float* input, float* result, int N) {
	// Write code here
	*result = d_max;
}


extern "C" void solve(const float* input, float* result, int N) {
	int threads = 256;
	int blocks = (N + threads - 1) / threads;
	float neg_inf = -FLT_MAX;
	
	int currentBlock = blocks;
	float *d_nums_reduced = nullptr;

	while(currentBlock > 1) {
		int nextBlock = (N + 2*threads-1)/(2*threads);
		int reduced_byteSize = nextBlock * sizeof(int);
		cudaMalloc(&d_nums_reduced, reduced_byteSize);
		gpuReduction<<<nextBlock, blocks, blocks * sizeof(float)>>>(input, d_nums_reduced, N);
		cudaDeviceSynchronize();
		input = d_nums_reduced;
		N = nextBlock;
		currentBlock = nextBlock;
	}




	cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
	max_kernel<<<1, 1>>>(input, result, N);
	cudaDeviceSynchronize();
}
