#include <cuda_runtime.h>
#include <math.h>

__device__ float d_sum;

__global__ void sum_kernel(const float* input, int N) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;

	extern __shared__ float shmem[] ;

	if(tid < N) {
		shmem[tx] = input[tid];
	}

	else {
		shmem[tx] = 0.0f;
	}

	__syncthreads();

	for(int stride = blockDim.x/2; stride > 0;stride /= 2) {
		if(tx < stride) {
			shmem[tx] += shmem[tx + stride];
		}

		__syncthreads();
	}

	if(tx == 0) {
		atomicAdd(&d_sum, shmem[0]);        
	}
}

__global__ void l1_normalize_kernel(const float* input, float* output, int N) {
  
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < N) {
		output[tid] = input[tid]/d_sum;
	}
}

extern "C" void solve(const float* input, float* output, int N) {
	int threads = 256;
	int blocks = (N + threads - 1) / threads;
	float zero = 0.0f;
	cudaMemcpyToSymbol(d_sum, &zero, sizeof(float));
	sum_kernel<<<blocks, threads, threads*sizeof(float)>>>(input, N);
	cudaDeviceSynchronize();
	l1_normalize_kernel<<<blocks, threads>>>(input, output, N);
	cudaDeviceSynchronize();
}
