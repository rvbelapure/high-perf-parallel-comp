#include <stdlib.h>
#include <stdio.h>

#include "cuda_utils.h"
#include "timer.c"

typedef float dtype;


__global__ 
void matTrans(dtype* AT, dtype* A, int N)  {
	/* Fill your code here */

}

__global__
void warmup() {
	for(int i = 0 ; i < 1000 ; i++);
}

void
parseArg (int argc, char** argv, int* N)
{
	if(argc == 2) {
		*N = atoi (argv[1]);
		assert (*N > 0);
	} else {
		fprintf (stderr, "usage: %s <N>\n", argv[0]);
		exit (EXIT_FAILURE);
	}
}


void
initArr (dtype* in, int N)
{
	int i;

	for(i = 0; i < N; i++) {
		in[i] = (dtype) rand () / RAND_MAX;
	}
}

void
cpuTranspose (dtype* A, dtype* AT, int N)
{
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			AT[j * N + i] = A[i * N + j];
		}
	}
}

int
cmpArr (dtype* a, dtype* b, int N)
{
	int cnt, i;

	cnt = 0;
	for(i = 0; i < N; i++) {
		if(abs(a[i] - b[i]) > 1e-6) cnt++;
	}

	return cnt;
}



void
gpuTranspose (dtype* A, dtype* AT, int N)
{
  struct stopwatch_t* timer = NULL;
  long double t_gpu;

  void (*kernel)(dtype* , dtype* , int );		// kernel pointer - change the association to determine which kernel to launch
  kernel = &matTrans;

  /* Now we have A as input array and AT as the output array on host side. 
     N is the length of side for square matrix N * N */

  /* 1. allocate device input output arrays */
  dtype *d_A, *d_AT;
  CUDA_CHECK_ERROR(cudaMalloc((void **) &d_A, N * N * sizeof(dtype)));
  CUDA_CHECK_ERROR(cudaMalloc((void **) &d_AT, N * N * sizeof(dtype)));

  /* 2. Fill the device input array */
  CUDA_CHECK_ERROR(cudaMemcpy(d_A, A, N * N * sizeof(dtype), cudaMemcpyHostToDevice));

  /* 3. Calculate gridDim and blockDim here */
  dim3 blkDim, grdDim;
	
  /* 4. Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

  /* 5. warmup */
  cudaFree(0);
  warmup<<<25,25>>>();
  cudaThreadSynchronize ();

  stopwatch_start (timer);
  /* 6. run your kernel here */
  kernel<<<grdDim, blkDim>>>(d_AT, d_A, N);
  cudaThreadSynchronize ();
  t_gpu = stopwatch_stop (timer);
  fprintf (stderr, "GPU transpose: %Lg secs ==> %Lg billion elements/second\n",
           t_gpu, (N * N) / t_gpu * 1e-9 );

  /* 7. copy the answer back to host array for further checking */
  CUDA_CHECK_ERROR( cudaMemcpy( AT, d_AT, N * N * sizeof(dtype), cudaMemcpyDeviceToHost));

  /* 8. Free the device memory */
  CUDA_CHECK_ERROR( cudaFree(d_A));
  CUDA_CHECK_ERROR( cudaFree(d_AT));
}

int 
main(int argc, char** argv)
{
  /* variables */
	dtype *A, *ATgpu, *ATcpu;
  int err;

	int N;

  struct stopwatch_t* timer = NULL;
  long double t_cpu;


	N = -1;
	parseArg (argc, argv, &N);

  /* input and output matrices on host */
  /* output */
  ATcpu = (dtype*) malloc (N * N * sizeof (dtype));
  ATgpu = (dtype*) malloc (N * N * sizeof (dtype));

  /* input */
  A = (dtype*) malloc (N * N * sizeof (dtype));

	initArr (A, N * N);

	/* GPU transpose kernel */
	gpuTranspose (A, ATgpu, N);

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

	stopwatch_start (timer);
  /* compute reference array */
	cpuTranspose (A, ATcpu, N);
  t_cpu = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute CPU transpose kernel: %Lg secs\n",
           t_cpu);

  /* check correctness */
	err = cmpArr (ATgpu, ATcpu, N * N);
	if(err) {
		fprintf (stderr, "Transpose failed: %d\n", err);
	} else {
		fprintf (stderr, "Transpose successful\n");
	}

	free (A);
	free (ATgpu);
	free (ATcpu);

  return 0;
}
