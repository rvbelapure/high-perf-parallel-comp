#include "driver.h"
#include "mm.h"
#include "cuda_utils.h"

void
initCudaArray (dtype **d_A, dtype *h_A, unsigned int N)
{
	CUDA_CHECK_ERROR (cudaMalloc ((void**) d_A, N * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMemcpy (*d_A, h_A, N * sizeof (dtype),
																cudaMemcpyHostToDevice));
}


__global__
void
mmSharedKernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	/* block indices */
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;

	/* thread indices */
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	/* row  index of first sub-block of matrix A processed by this thread block */
	int aStart = N * (BLOCK_SIZE * bidy);
	/* row  index of last sub-block of matrix A processed by this thread block */
	int aEnd   = aStart + N - 1;
	/* increment size for sub-block of matrix A */
	int aInc = BLOCK_SIZE;

	/* col index of first sub-blcok of matrx B processed by this thread block */
	int bStart = BLOCK_SIZE * bidx;
	/* last sub block is not needed since it'll have 1-on-1 match to A */
	/* increment size for sub-block of matrix B */
	int bInc = BLOCK_SIZE * N;

	/* temporary variable for accummulating the partial results */
	float cSub = 0;

	/* Loop over the sub-matrices of A and B */
	for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
		/* declaration of shared memory for storing sub-block of A */
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		/* declaration of shared memory for storing sub-block of B */
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		/* load the matrices from memory to shared memory */
		As[tidy][tidx] = A[a + N * tidy + tidx];
		Bs[tidy][tidx] = B[b + N * tidy + tidx];
		__syncthreads();

		/* multiply the two matrices together */
		/* one thread per element of C */
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
			cSub += As[tidy][k] * Bs[k][tidx];

		/* synchornize before loading next sub-blocks */
		__syncthreads();
	}

	/* write back the results */
	int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
	C[c + N * tidy + tidx] = cSub;

}
void
mmShared (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE);	

	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}



__global__
void
mmNaiveKernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	int i;
	dtype sum;
	int gidx = threadIdx.x + blockIdx.x * blockDim.x; /* column (j) */
	int gidy = threadIdx.y + blockIdx.y * blockDim.y; /* row (i) */
	int gid = gidx + gidy * N;

	sum = 0.0;
	for(i = 0; i < N; i++) {
		sum += A[gidy * N + i] * B[i * N + gidx];
	}
	C[gid] = sum;
}
void
mmNaive (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE);	


	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();


	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}


__global__
void
mmShared2Kernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	/* insert your code here */
}
void
mmShared2 (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE / 2);	

	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}


__global__
void
mmShared4Kernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	/* insert your code here */
}
void
mmShared4 (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE / 4);	

	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}



__global__
void
mmShared8Kernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	/* insert your code here */
}
void
mmShared8 (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE / 8);	

	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}




void
cudaMM (dtype *A, dtype* B, dtype* C, unsigned int N, unsigned int OPT, dtype* h_C)
{
	cudaEvent_t start, stop;
	float elapsedTime;

	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));

	fprintf (stderr, "Executing test case [%d]\n", OPT);
	fprintf (stderr, "[1]: Naive | [2]: shared memory| [3]: SM 2 per thread | [4]: SM 4 per thread | [5]: SM 8 per thread | \n");

	
	CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	/* execute kernel */
	switch (OPT) {
		case 1:
			mmNaive (A, B, C, N);	
			break;
		case 2:
			mmShared (A, B, C, N);	
			break;
		case 3:
			mmShared2 (A, B, C, N);	
			break;
		case 4:
			mmShared4 (A, B, C, N);	
			break;
		case 5:
			mmShared8 (A, B, C, N);	
			break;
		default:
			mmNaive (A, B, C, N);	
	} 
	CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
	CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
	CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
	elapsedTime = elapsedTime / 5;

	CUDA_CHECK_ERROR (cudaMemcpy (h_C, C, N * N * sizeof (dtype), 
																cudaMemcpyDeviceToHost));

	fprintf (stderr, "Execution time: %f ms\n", elapsedTime);
	fprintf (stderr, "Equivalent performance: %f GFLOP/s\n", 
						1e-6 * 2 * N * N * N / elapsedTime );

	CUDA_CHECK_ERROR (cudaEventDestroy (start));
	CUDA_CHECK_ERROR (cudaEventDestroy (stop));

}


