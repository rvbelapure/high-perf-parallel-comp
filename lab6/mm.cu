#include "driver.h"
#include "mm.h"
#include "cuda_utils.h"

#define MY_BLOCK_SIZE 32

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
	float cSub1 = 0, cSub2 = 0;

	/* Loop over the sub-matrices of A and B */
	for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
		/* declaration of shared memory for storing sub-block of A */
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		/* declaration of shared memory for storing sub-block of B */
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		/* load the matrices from memory to shared memory */
		As[tidy][tidx] = A[a + N * tidy + tidx];
		Bs[tidy][tidx] = B[b + N * tidy + tidx];
		As[tidy + BLOCK_SIZE/2][tidx] = A[a + N * (tidy+BLOCK_SIZE/2) + tidx];
		Bs[tidy + BLOCK_SIZE/2][tidx] = B[b + N * (tidy+BLOCK_SIZE/2) + tidx];
		__syncthreads();

		/* multiply the two matrices together */
		/* one thread per element of C */
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			cSub1 += As[tidy][k] * Bs[k][tidx];
			cSub2 += As[tidy+BLOCK_SIZE/2][k] * Bs[k][tidx];
		}

		/* synchornize before loading next sub-blocks */
		__syncthreads();
	}

	/* write back the results */
	int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
	C[c + N * tidy + tidx] = cSub1;
	C[c + N * (tidy + BLOCK_SIZE/2) + tidx] = cSub2;
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
	float cSub1 = 0, cSub2 = 0, cSub3 = 0, cSub4 = 0;

	/* Loop over the sub-matrices of A and B */
	for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
		/* declaration of shared memory for storing sub-block of A */
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		/* declaration of shared memory for storing sub-block of B */
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		/* load the matrices from memory to shared memory */
		As[tidy][tidx] = A[a + N * tidy + tidx];
		Bs[tidy][tidx] = B[b + N * tidy + tidx];

		As[tidy + BLOCK_SIZE/4][tidx] = A[a + N * (tidy+BLOCK_SIZE/4) + tidx];
		Bs[tidy + BLOCK_SIZE/4][tidx] = B[b + N * (tidy+BLOCK_SIZE/4) + tidx];

		As[tidy + BLOCK_SIZE/2][tidx] = A[a + N * (tidy+BLOCK_SIZE/2) + tidx];
		Bs[tidy + BLOCK_SIZE/2][tidx] = B[b + N * (tidy+BLOCK_SIZE/2) + tidx];

		As[tidy + 3*BLOCK_SIZE/4][tidx] = A[a + N * (tidy+3*BLOCK_SIZE/4) + tidx];
		Bs[tidy + 3*BLOCK_SIZE/4][tidx] = B[b + N * (tidy+ 3*BLOCK_SIZE/4) + tidx];

		__syncthreads();

		/* multiply the two matrices together */
		/* one thread per element of C */
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			cSub1 += As[tidy][k] * Bs[k][tidx];
			cSub2 += As[tidy+BLOCK_SIZE/4][k] * Bs[k][tidx];
			cSub3 += As[tidy+BLOCK_SIZE/2][k] * Bs[k][tidx];
			cSub4 += As[tidy+3*BLOCK_SIZE/4][k] * Bs[k][tidx];
		}

		/* synchornize before loading next sub-blocks */
		__syncthreads();
	}

	/* write back the results */
	int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
	C[c + N * tidy + tidx] = cSub1;
	C[c + N * (tidy + BLOCK_SIZE/4) + tidx] = cSub2;
	C[c + N * (tidy + BLOCK_SIZE/2) + tidx] = cSub3;
	C[c + N * (tidy + 3*BLOCK_SIZE/4) + tidx] = cSub4;

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
	float cSub1 = 0, cSub2 = 0, cSub3 = 0, cSub4 = 0, cSub5 = 0, cSub6 = 0, cSub7 = 0, cSub8 = 0;

	/* Loop over the sub-matrices of A and B */
	for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
		/* declaration of shared memory for storing sub-block of A */
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		/* declaration of shared memory for storing sub-block of B */
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		/* load the matrices from memory to shared memory */
		As[tidy][tidx] = A[a + N * tidy + tidx];
		Bs[tidy][tidx] = B[b + N * tidy + tidx];

		As[tidy + BLOCK_SIZE/8][tidx] = A[a + N * (tidy+BLOCK_SIZE/8) + tidx];
		Bs[tidy + BLOCK_SIZE/8][tidx] = B[b + N * (tidy+BLOCK_SIZE/8) + tidx];

		As[tidy + 2*BLOCK_SIZE/8][tidx] = A[a + N * (tidy+2*BLOCK_SIZE/8) + tidx];
		Bs[tidy + 2*BLOCK_SIZE/8][tidx] = B[b + N * (tidy+2*BLOCK_SIZE/8) + tidx];

		As[tidy + 3*BLOCK_SIZE/8][tidx] = A[a + N * (tidy+ 3*BLOCK_SIZE/8) + tidx];
		Bs[tidy + 3*BLOCK_SIZE/8][tidx] = B[b + N * (tidy+ 3*BLOCK_SIZE/8) + tidx];

		As[tidy + 4*BLOCK_SIZE/8][tidx] = A[a + N * (tidy+ 4*BLOCK_SIZE/8) + tidx];
		Bs[tidy + 4*BLOCK_SIZE/8][tidx] = B[b + N * (tidy+ 4*BLOCK_SIZE/8) + tidx];

		As[tidy + 5*BLOCK_SIZE/8][tidx] = A[a + N * (tidy+ 5*BLOCK_SIZE/8) + tidx];
		Bs[tidy + 5*BLOCK_SIZE/8][tidx] = B[b + N * (tidy+ 5*BLOCK_SIZE/8) + tidx];

		As[tidy + 6*BLOCK_SIZE/8][tidx] = A[a + N * (tidy+ 6*BLOCK_SIZE/8) + tidx];
		Bs[tidy + 6*BLOCK_SIZE/8][tidx] = B[b + N * (tidy+ 6*BLOCK_SIZE/8) + tidx];

		As[tidy + 7*BLOCK_SIZE/8][tidx] = A[a + N * (tidy+ 7*BLOCK_SIZE/8) + tidx];
		Bs[tidy + 7*BLOCK_SIZE/8][tidx] = B[b + N * (tidy+ 7*BLOCK_SIZE/8) + tidx];

		__syncthreads();

		/* multiply the two matrices together */
		/* one thread per element of C */
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			cSub1 += As[tidy][k] * Bs[k][tidx];
			cSub2 += As[tidy+BLOCK_SIZE/8][k] * Bs[k][tidx];
			cSub3 += As[tidy+2*BLOCK_SIZE/8][k] * Bs[k][tidx];
			cSub4 += As[tidy+3*BLOCK_SIZE/8][k] * Bs[k][tidx];
			cSub5 += As[tidy+4*BLOCK_SIZE/8][k] * Bs[k][tidx];
			cSub6 += As[tidy+5*BLOCK_SIZE/8][k] * Bs[k][tidx];
			cSub7 += As[tidy+6*BLOCK_SIZE/8][k] * Bs[k][tidx];
			cSub8 += As[tidy+7*BLOCK_SIZE/8][k] * Bs[k][tidx];
		}

		/* synchornize before loading next sub-blocks */
		__syncthreads();
	}

	/* write back the results */
	int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
	C[c + N * tidy + tidx] = cSub1;
	C[c + N * (tidy + BLOCK_SIZE/8) + tidx] = cSub2;
	C[c + N * (tidy + 2*BLOCK_SIZE/8) + tidx] = cSub3;
	C[c + N * (tidy + 3*BLOCK_SIZE/8) + tidx] = cSub4;
	C[c + N * (tidy + 4*BLOCK_SIZE/8) + tidx] = cSub5;
	C[c + N * (tidy + 5*BLOCK_SIZE/8) + tidx] = cSub6;
	C[c + N * (tidy + 6*BLOCK_SIZE/8) + tidx] = cSub7;
	C[c + N * (tidy + 7*BLOCK_SIZE/8) + tidx] = cSub8;

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

__global__
void
mmMyOwnKernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	/* insert your code here */
	/* block indices */
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;

	/* thread indices */
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	/* row  index of first sub-block of matrix A processed by this thread block */
	int aStart = N * (MY_BLOCK_SIZE * bidy);
	/* row  index of last sub-block of matrix A processed by this thread block */
	int aEnd   = aStart + N - 1;
	/* increment size for sub-block of matrix A */
	int aInc = MY_BLOCK_SIZE;

	/* col index of first sub-blcok of matrx B processed by this thread block */
	int bStart = MY_BLOCK_SIZE * bidx;
	/* last sub block is not needed since it'll have 1-on-1 match to A */
	/* increment size for sub-block of matrix B */
	int bInc = MY_BLOCK_SIZE * N;

	/* temporary variable for accummulating the partial results */
	float cSub[16] = {0};

	/* Loop over the sub-matrices of A and B */
	for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
		/* declaration of shared memory for storing sub-block of A */
		__shared__ float As[MY_BLOCK_SIZE][MY_BLOCK_SIZE];

		/* declaration of shared memory for storing sub-block of B */
		__shared__ float Bs[MY_BLOCK_SIZE][MY_BLOCK_SIZE];

		/* load the matrices from memory to shared memory */
		#pragma unroll
		for(int i=0; i<8; i++){
			As[tidy + i*4][tidx] = A[a + N * (tidy + i * 4) + tidx];
			Bs[tidy + i*4][tidx] = B[b + N * (tidy + i * 4) + tidx];

			As[tidy + i*4][tidx + 16] = A[a + N * (tidy + i * 4) + tidx + 16];
			Bs[tidy + i*4][tidx + 16] = B[b + N * (tidy + i * 4) + tidx + 16];
		}

		__syncthreads();

		/* multiply the two matrices together */
		/* one thread per element of C */
		#pragma unroll
		for (int k = 0; k < MY_BLOCK_SIZE; ++k)
		{
			cSub[0] += As[tidy][k] * Bs[k][tidx];
			cSub[1] += As[tidy + 4][k] * Bs[k][tidx];
			cSub[2] += As[tidy + 8][k] * Bs[k][tidx];
			cSub[3] += As[tidy + 12][k] * Bs[k][tidx];
			cSub[4] += As[tidy + 16][k] * Bs[k][tidx];
			cSub[5] += As[tidy + 20][k] * Bs[k][tidx];
			cSub[6] += As[tidy + 24][k] * Bs[k][tidx];
			cSub[7] += As[tidy + 28][k] * Bs[k][tidx];

			cSub[8] += As[tidy][k] * Bs[k][tidx + 16];
			cSub[9] += As[tidy + 4][k] * Bs[k][tidx + 16];
			cSub[10] += As[tidy + 8][k] * Bs[k][tidx + 16];
			cSub[11] += As[tidy + 12][k] * Bs[k][tidx + 16];                       
			cSub[12] += As[tidy + 16][k] * Bs[k][tidx + 16];  
			cSub[13] += As[tidy + 20][k] * Bs[k][tidx + 16];                        
			cSub[14] += As[tidy + 24][k] * Bs[k][tidx + 16];
			cSub[15] += As[tidy + 28][k] * Bs[k][tidx + 16];
		}

		/* synchornize before loading next sub-blocks */
		__syncthreads();
	}

	/* write back the results */
	int c = N * MY_BLOCK_SIZE * bidy + MY_BLOCK_SIZE * bidx;
	C[c + N * tidy + tidx] = cSub[0];
	C[c + N * (tidy + 4) + tidx] = cSub[1];
	C[c + N * (tidy + 8) + tidx] = cSub[2];
	C[c + N * (tidy + 12) + tidx] = cSub[3];
	C[c + N * (tidy + 16) + tidx] = cSub[4];
	C[c + N * (tidy + 20) + tidx] = cSub[5];
	C[c + N * (tidy + 24) + tidx] = cSub[6];
	C[c + N * (tidy + 28) + tidx] = cSub[7];

	C[c + N * tidy + tidx + 16] = cSub[8];
	C[c + N * (tidy + 4) + tidx + 16] = cSub[9];
	C[c + N * (tidy + 8) + tidx + 16] = cSub[10];
	C[c + N * (tidy + 12) + tidx + 16] = cSub[11];
	C[c + N * (tidy + 16) + tidx + 16] = cSub[12];
	C[c + N * (tidy + 20) + tidx + 16] = cSub[13];
	C[c + N * (tidy + 24) + tidx + 16] = cSub[14];
	C[c + N * (tidy + 28) + tidx + 16] = cSub[15];
}
void
mmMyOwn (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;

	nBlocks = (N + MY_BLOCK_SIZE - 1) / MY_BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (MY_BLOCK_SIZE / 2, MY_BLOCK_SIZE / 8);	

	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
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
	fprintf (stderr, "[1]: Naive | [2]: shared memory| [3]: SM 2 per thread | [4]: SM 4 per thread | [5]: SM 8 per thread | [6]: my own implementation \n");

	
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
		case 6:
			mmMyOwn (A, B, C, N);
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


