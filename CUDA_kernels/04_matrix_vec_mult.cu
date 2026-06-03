#include <cuda_runtime.h>

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
	// Write code here
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < M) {
		//yi = Aij*xj
		float sum = 0.0f;
		for(int col = 0;col < N;col++) {
			sum += A[row * N + col] * x[col];
		}

		y[row] = sum;
			
	}
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    dim3 threads(256);
    dim3 blocks((M + 255) / 256);
    gemv_kernel<<<blocks, threads>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}

