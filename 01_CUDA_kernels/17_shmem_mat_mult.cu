#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  // Write code here

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int global_col = blockIdx.x * TILE_DIM + tx;
	int global_row = blockIdx.y * TILE_DIM + ty;

	__shared__ float a_shared[TILE_DIM][TILE_DIM];
	__shared__ float b_shared[TILE_DIM][TILE_DIM];

	float sum = 0.0f;
	for(int tile = 0; tile < (K + TILE_DIM - 1)/TILE_DIM; tile++) {

		int a_col = tile * TILE_DIM + tx;
		int b_row = tile * TILE_DIM + ty;

		//load A into shared memory
		if(global_row < M && a_col < K)	a_shared[ty][tx] = A[global_row * K + a_col];
		else a_shared[ty][tx] = 0.0f;

		//load B into shared memory
		if(b_row < K && global_col < N) b_shared[ty][tx] = B[b_row * N + global_col];
		else b_shared[ty][tx] = 0.0f;

		__syncthreads();

		for(int k = 0;k < TILE_DIM;k++) {
			sum += a_shared[ty][k] * b_shared[k][tx];
		}	
		
		__syncthreads();
	}

	//Store to the output
	if(global_row < M && global_col < N)
  C[global_row * N + global_col] = sum;
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
	dim3 threads(TILE_DIM, TILE_DIM);
	dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
	tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
	cudaDeviceSynchronize();
}