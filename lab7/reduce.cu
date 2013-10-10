#include "driver.h"
#include "reduce.h"
#include "cuda_utils.h"

dtype
reduceCpu (dtype* h_A, unsigned int N)
{
  int i;
  dtype ans;

  ans = (dtype) 0.0;
  for(i = 0; i < N; i++) {
    ans += h_A[i];
  }

  return ans;
}

__global__ void 
reduceNaiveKernel (dtype* In, dtype *Out, unsigned int N)
{
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride;
	

	/* load data to buffer */
	if(tid < N) {
		buffer[threadIdx.x] = In[tid];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();

	/* reduce in shared memory */
	for(stride = 1; stride < blockDim.x; stride *= 2) {
		if(threadIdx.x % (stride * 2) == 0) {
			buffer[threadIdx.x] += buffer[threadIdx.x + stride];
		}
		__syncthreads ();
	}

	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}
}

dtype
reduceNaive (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;
	

	nThreads = N;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceNaiveKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceNonDivergeKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride;
	

	/* load data to buffer */
	if(tid < N) {
		buffer[threadIdx.x] = In[tid];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();

	/* reduce in shared memory */
	int threadcount = blockDim.x / 2;
	for(stride = 1; stride < blockDim.x; stride *= 2) {
		if(threadIdx.x < threadcount)
		{
			buffer[threadIdx.x * stride * 2] += buffer[threadIdx.x * stride * 2 + stride];
		}
		threadcount /= 2;
		__syncthreads ();
	}

	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}

}



dtype
reduceNonDiverge (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = N;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceNonDivergeKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceSeqAddKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	/* Replicate the access pattern as shown the lecture slides for version 3 */
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride;
	

	/* load data to buffer */
	if(tid < N) {
		buffer[threadIdx.x] = In[tid];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();

	/* reduce in shared memory */
	int threadcount = blockDim.x / 2;
#pragma unroll
	for(stride = 1; stride < BS ; stride *= 2) {
		if(threadIdx.x < threadcount)
		{
			buffer[threadIdx.x] += buffer[threadIdx.x + threadcount];
		}
		threadcount /= 2;
		__syncthreads ();
	}

	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}

}



dtype
reduceSeqAdd (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = N;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceSeqAddKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceFirstAddKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	/* As it can be seen from `reduceSeqAdd`, the total number of threads
		 have been halved */
	/* Thus, you need to load 2 elements from the global memory, add them, and
		 then store the sum in the shared memory before reduction over the shared
		 memory occurs */
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride;
	

	int threadcount = blockDim.x;
	/* load data to buffer */
	if(tid < N) {
		dtype ele = 0.0;
		if(tid + threadcount < N)
			ele = In[tid + threadcount];

		buffer[threadIdx.x] = In[tid] + ele;

	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();

	threadcount /= 2;
	/* reduce in shared memory */
#pragma unroll
	for(stride = 1; stride < BS ; stride *= 2) {
		if(threadIdx.x < threadcount)
		{
			buffer[threadIdx.x] += buffer[threadIdx.x + threadcount];
		}
		threadcount /= 2;
		__syncthreads ();
	}

	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}

}



dtype
reduceFirstAdd (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = (N + 1) / 2;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceFirstAddKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceUnrollLastKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	/* unroll the loop when there are fewer than 32 threads working */
}



dtype
reduceUnrollLast (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = (N + 1) / 2;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceUnrollLastKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}


__global__ void 
reduceUnrollAllKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	/* do a complete unrolling using #define or -D compiler option to specify 
		 the thread block size */
}



dtype
reduceUnrollAll (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = (N + 1) / 2;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceUnrollAllKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}



__global__ void 
reduceMultAddKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	/* Instead of just adding 2 elements in the beginning, try adding more 
		 before reducing the partial sums over the shared memory */
}



dtype
reduceMultAdd (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = (N + 1) / 32;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceMultAddKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}





void
initCudaArray (dtype **d_A, dtype *h_A, unsigned int N)
{
	CUDA_CHECK_ERROR (cudaMalloc ((void**) d_A, N * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMemcpy (*d_A, h_A, N * sizeof (dtype),
																cudaMemcpyHostToDevice));
}

void
cudaReduction (dtype *A, unsigned int N, unsigned int OPT, dtype *ret)
{
	dtype *h_Out, *d_Out;
	unsigned int nBlocks;

	cudaEvent_t start, stop;
	float elapsedTime;

	dtype ans;

	nBlocks = (N + BS - 1) / BS;
	h_Out = (dtype*) malloc (nBlocks * sizeof (dtype));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_Out, nBlocks * sizeof (dtype)));
	
	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));

	fprintf (stderr, "Executing test case [%d]\n", OPT);
	fprintf (stderr, "[1]: Naive | [2]: Non-divergent | [3]: Sequential Add. | [4]: First add | [5]: Unroll last warp | [6]: Complete unroll | [7] Multiple Adds\n");

	
	CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	/* execute kernel */
	switch (OPT) {
		case 1:
			ans = reduceNaive (A, d_Out, h_Out, N);	
			break;
		case 2:
			ans = reduceNonDiverge (A, d_Out, h_Out, N);	
			break;
		case 3:
			ans = reduceSeqAdd (A, d_Out, h_Out, N);	
			break;
		case 4:
			ans = reduceFirstAdd (A, d_Out, h_Out, N);	
			break;
		case 5:
			ans = reduceUnrollLast (A, d_Out, h_Out, N);	
			break;
		case 6:
			ans = reduceUnrollAll (A, d_Out, h_Out, N);	
			break;
		case 7:
			ans = reduceMultAdd (A, d_Out, h_Out, N);	
			break;
		default:
			ans = reduceNaive (A, d_Out, h_Out, N);	
	} 
	CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
	CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
	CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
	elapsedTime = elapsedTime / NUM_ITER;


	fprintf (stderr, "Execution time: %f ms\n", elapsedTime);
	fprintf (stderr, "Equivalent performance: %f GB/s\n", 
						(N * sizeof (dtype) / elapsedTime) * 1e-6);

	CUDA_CHECK_ERROR (cudaEventDestroy (start));
	CUDA_CHECK_ERROR (cudaEventDestroy (stop));

	free (h_Out);
	CUDA_CHECK_ERROR (cudaFree (d_Out));

	*ret = ans;	
}


