__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
	// TODO: compute C = A * B for row-major matrices
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j =  blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < ) {
		float sum = 0.0f;
		


	} 


}


//A is MXK
//B is KXN
//C is MXN

for(int i = 0;i<M;i++) {
	for(int j = 0;j<N;j++) {
		float sum = 0;
		for(int k = 0;k<K;k++) {
			//cij = aik*bkj;
			sum += a[i*K + k] * b[k*N + j];
		}
		c[i*N + j] = sum;
	}
}