#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {

	int tx = threadIdx.x;
	int row = blockIdx.x;

	extern __shared__ float shmem[];

	float *reduction = shmem;

	// Find mean
	float local_sum = 0.0f;
	for(int i = tx; i < N; i += blockDim.x) {
		local_sum += input[row * N + i];
	}

	reduction[tx] = local_sum;
	__syncthreads();

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if(tx < stride) {
			reduction[tx] += reduction[tx + stride];
		}
		__syncthreads();
	}

	float mean = reduction[0] / N;
	__syncthreads();

	// Find variance
	float local_var = 0.0f;
	for(int i = tx; i < N; i += blockDim.x) {
		float diff = input[row * N + i] - mean;
		local_var += diff * diff;
	}

	reduction[tx] = local_var;
	__syncthreads();

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if(tx < stride) {
			reduction[tx] += reduction[tx + stride];
		}
		__syncthreads();
	}

	float variance = reduction[0] / N;
	float inv_std = rsqrtf(variance + eps);
	__syncthreads();

	// Normalize and affine transform
	for(int i = tx; i < N; i += blockDim.x) {
		float x = input[row * N + i];
		float norm = (x - mean) * inv_std;
		output[row * N + i] = norm * gamma[i] + beta[i];
	}
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {

	int threads = 256;
	dim3 blocks(M);

	layer_norm_kernel<<<blocks, threads, threads * sizeof(float)>>>(input,gamma,beta,	output,	M,N,eps);

	cudaDeviceSynchronize();
}