#include <stdlib.h>
#include <stdio.h>

#include "cuda_utils.h"
#include "timer.c"

#define TILE_DIM   32	// make side of matrix multiple of 32
#define BLOCK_ROWS 8

#define DEBUG 0


typedef float dtype;

__global__
void matTrans(dtype* AT, dtype* A, int N)  {
	int horloc = blockIdx.x * TILE_DIM + threadIdx.x;
	int verloc = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
		AT[horloc * width + (verloc + j)] = A[(verloc + j)*width + horloc];
}

__global__
void warmup() {
	for(int i = 0 ; i < 1000 ; i++);
}

void printArray(int N, dtype *A, char *message)
{
	printf("%s : \n", message);
	for(int i = 0; i < N ; i++)
	{
		for(int j = 0 ; j < N ; j++)
			printf("%ld ",A[i * N + j]);
		printf("\n");
	}
	printf("---------------------------\n");
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

  #if DEBUG
  printArray(N, A, "input");
  #endif

  /* Now we have A as input array and AT as the output array on host side. 
     N is the length of side for square matrix N * N */
  /* 0. As per our algorithm, we have to make sure that side of the matrix is multiple of our TILE_DIM.
     Thus, we pad the matrix with extra elements */
  int tiled_size;
  if (N % TILE_DIM == 0)
	  tiled_size = N;
  else
  {
	  tiled_size = ((N / TILE_DIM) + 1) * TILE_DIM;
  }
  dtype * padded_input = (dtype *) malloc( tiled_size * tiled_size * sizeof(dtype));
  // we can not use memcpy as we should not copy into padded region
  for(int i = 0 ; i < N ; i++)
	  for(int j = 0 ; j < N ; j++)
		  padded_input[i * tiled_size + j] = A[i * N + j];	
  #if DEBUG
  printArray(tiled_size, padded_input, "padded input");
  #endif


  /* 1. allocate device input output arrays */
  dtype *d_A, *d_AT;
  CUDA_CHECK_ERROR(cudaMalloc((void **) &d_A, tiled_size * tiled_size * sizeof(dtype)));
  CUDA_CHECK_ERROR(cudaMalloc((void **) &d_AT, tiled_size * tiled_size * sizeof(dtype)));

  /* 2. Fill the device input array */
  CUDA_CHECK_ERROR(cudaMemcpy(d_A, padded_input, tiled_size * tiled_size * sizeof(dtype), cudaMemcpyHostToDevice));

  /* 3. Calculate gridDim and blockDim here */
  dim3 grdDim( tiled_size / TILE_DIM, tiled_size / TILE_DIM, 1);
  dim3 blkDim( TILE_DIM, BLOCK_ROWS, 1);
	
  /* 4. Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

  /* 5. warmup */
  cudaFree(0);
  warmup<<<25,25>>>();
  cudaThreadSynchronize ();

  stopwatch_start (timer);
  /* 6. run your kernel here */
  kernel<<<grdDim, blkDim>>>(d_AT, d_A, tiled_size);
  cudaThreadSynchronize ();
  t_gpu = stopwatch_stop (timer);
  fprintf (stderr, "GPU transpose: %Lg secs ==> %Lg billion elements/second\n",
           t_gpu, (N * N) / t_gpu * 1e-9 );

  /* 7. copy the answer back to host array for further checking */
  CUDA_CHECK_ERROR( cudaMemcpy( padded_input, d_AT, tiled_size * tiled_size * sizeof(dtype), cudaMemcpyDeviceToHost));
  for(int i = 0 ; i < N ; i++)
	  for(int j = 0 ; j < N ; j++)
		  AT[i * N + j] = padded_input[i * tiled_size + j];
  #if DEBUG
  printArray(tiled_size, padded_input, "padded output");
  printArray(N, AT, "output");
  #endif

  /* 8. Free the device memory */
  free(padded_input);
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
