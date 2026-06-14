#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
	
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
		atomicAdd(result, shmem[0]);
	}

}

extern "C" void solve(const float* input, float* result, int N) {
	int threads = 256;
	int blocks = (N + threads - 1) / threads;
	cudaMemset(result, 0, sizeof(float));
	sum_kernel<<<blocks, threads, threads*sizeof(float)>>>(input, result, N);
	cudaDeviceSynchronize();
}
