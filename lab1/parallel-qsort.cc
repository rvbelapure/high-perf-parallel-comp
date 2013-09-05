/**
 *  \file parallel-qsort.cc
 *
 *  \brief Implement your parallel quicksort using Cilk Plus in this
 *  file, given an initial sequential implementation.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "sort.hh"
#include <cilk/reducer_opadd.h>

#define DEBUG 0


void printArray(int N, keytype* A)
{
	printf("printarray : ");
	for(int i = 0 ; i < N ; i++)
		printf("%ld ",A[i]);
	printf("\n");
}

keytype findMedian(keytype a, keytype b, keytype c)
{
	if(((a < b) && (b < c)) || ((a > b) && (b > c)))
		return b;
	else if(((b < c) && (c < a)) || (b > c) && (c > a))
		return c;
	else
		return a;
}

/**
 *  Pivots the keys of A[0:N-1] around a given pivot value. The number
 *  of keys less than the pivot is returned in *p_n_lt; the number
 *  equal in *p_n_eq; and the number greater in *p_n_gt. The
 *  rearranged keys are stored back in A as follows:
 *
 * - The first *p_n_lt elements of A are all the keys less than the
 *   pivot. That is, they appear in A[0:(*p_n_lt)-1].
 *
 * - The next *p_n_eq elements of A are all keys equal to the
 *   pivot. That is, they appear in A[(*p_n_lt):(*p_n_lt)+(*p_n_eq)-1].
 *
 * - The last *p_n_gt elements of A are all keys greater than the
 *   pivot. That is, they appear in
 *   A[(*p_n_lt)+(*p_n_eq):(*p_n_lt)+(*p_n_eq)+(*p_n_gt)-1].
 */

#if 0
/* Original implementation (with reducer added) */
void partition (keytype pivot, int N, keytype* A,
		int* p_n_lt, int* p_n_eq, int* p_n_gt)
{
  /* Count how many elements of A are less than (lt), equal to (eq),
     or greater than (gt) the pivot value. */
  int n_lt = 0, n_eq = 0, n_gt = 0;

  if(N < 500)
  {
	  for (int i = 0; i < N; ++i) {
	    if (A[i] < pivot) ++n_lt;
	    else if (A[i] == pivot) ++n_eq;
	    else ++n_gt;
	  }
  }
  else
  {

	  cilk::reducer_opadd<int> l;
	  cilk::reducer_opadd<int> e;
	  cilk::reducer_opadd<int> g;
	  _Cilk_for(int i = 0 ; i < N ; ++i)
	  {
		  if(A[i] < pivot) ++l;
		  else if (A[i] == pivot) ++e;
		  else ++g;
	  }

	  n_lt = l.get_value();
	  n_eq = e.get_value();
	  n_gt = g.get_value();
  }

  keytype* A_orig = newCopy (N, A);

  /* Next, rearrange A so that:
   *   A_lt == A[0:n_lt-1] == subset of A < pivot
   *   A_eq == A[n_lt:(n_lt+n_eq-1)] == subset of A == pivot
   *   A_gt == A[(n_lt+n_eq):(N-1)] == subset of A > pivot
   */
  int i_lt = 0; /* next open slot in A_lt */
  int i_eq = n_lt; /* next open slot in A_eq */
  int i_gt = n_lt + n_eq; /* next open slot in A_gt */
  for (int i = 0; i < N; ++i) {
    keytype ai = A_orig[i];
    if (ai < pivot)
      A[i_lt++] = ai;
    else if (ai > pivot)
      A[i_gt++] = ai;
    else
      A[i_eq++] = ai;
  }
  assert (i_lt == n_lt);
  assert (i_eq == (n_lt+n_eq));
  assert (i_gt == N);

  free (A_orig);

  if (p_n_lt) *p_n_lt = n_lt;
  if (p_n_eq) *p_n_eq = n_eq;
  if (p_n_gt) *p_n_gt = n_gt;
}

#endif

#if 0
/* Serial In-Place Partition */
void partition (keytype pivot, int N, keytype* A,
		int* p_n_lt, int* p_n_eq, int* p_n_gt)
{
	register int low = 0, high = N-1;
	int temp;

	for(low = 0 ; low <= high ; )
	{
		while(A[low] < pivot) low++;
		while(A[high] >= pivot) high--;
		if(low <= high)
		{
			temp = A[low];
			A[low] = A[high];
			A[high] = temp;
			low++;
			high--;
		}
	}

	if (p_n_lt) *p_n_lt = low;
	if (p_n_eq) *p_n_eq = 0;
	if (p_n_gt) *p_n_gt = N - low;
}
#endif

#if 1
/* Parallel in-place partitioning using divide and conquor */
void partition (keytype pivot, int N, keytype* A,
		int* p_n_lt, int* p_n_eq, int* p_n_gt)
{
	int low = 0, high = N-1;
	int temp;

	if(N <= 2500000)
	{
		#if DEBUG
		printf("before iter par : "); printArray(N, A);
		#endif

		for(low = 0 ; low <= high ; )
		{
			while((low < N)  && (A[low] < pivot))	low++;
			while((high >= 0) && (A[high] >= pivot)) high--;
			if((low < N) && (high >= 0) && (low <= high))
			{
				temp = A[low];
				A[low] = A[high];
				A[high] = temp;
				low++;
				high--;
			}
		}
		#if DEBUG
		printf("after iter par : "); printArray(N, A);
		#endif
	}
	else
	{
		#if DEBUG
		printf("recur par : "); printArray(N, A);
		#endif

		int n_lt1, n_eq1, n_gt1;
		int n_lt2, n_eq2, n_gt2;
		_Cilk_spawn partition(pivot, N/2, A, &n_lt1, &n_eq1, &n_gt1);
		partition(pivot, N - (N/2), A + (N/2), &n_lt2, &n_eq2, &n_gt2);
		_Cilk_sync;
		#if DEBUG
		printf("before blockswap : "); printArray(N, A);
		#endif

		/* Array looks as follows -->
		 * less1 | eq + gt | less2 | eq + gt
		 * we need to insert less2 after less1 so that final array looks like
		 * less1 | less2 | eq + gt | eq + gt
		 */
		int left_start = n_lt1, left_end = left_start + n_eq1 + n_gt1;
		int right_start = left_end, right_end = right_start + n_lt2;
		#if DEBUG
		printf("merging : left_start = %d, left_end = %d, right_start = %d, right_end = %d\n",
				left_start, left_end, right_start, right_end);
		#endif
		int i,j;
		if((n_lt2 > 0) && (n_gt1 > 0)) 
		{
			for(i = left_start, j = right_end - 1; i < j ; )
			{
				int temp = A[i];
				A[i] = A[j];
				A[j] = temp;
				i++;
				j--;
			}
		}
		
		low = n_lt1 + n_lt2;
		#if DEBUG
		printf("after blockswap : "); printArray(N, A);
		#endif
	}

	if (p_n_lt) *p_n_lt = low;
	if (p_n_eq) *p_n_eq = 0;
	if (p_n_gt) *p_n_gt = N - low;
}

#endif

void
quickSort (int N, keytype* A)
{
  const int G = 90; /* base case size, a tuning parameter */
  if (N < G)
    sequentialSort (N, A);
  else {
    // Choose pivot = median of 1st, Nth and N/2 th element
//      keytype pivot = findMedian(A[0], A[N], A[N/2]);
    keytype pivot = A[rand () % N];

    // Partition around the pivot. Upon completion, n_less, n_equal,
    // and n_greater should each be the number of keys less than,
    // equal to, or greater than the pivot, respectively. Moreover, the array
    int n_less = -1, n_equal = -1, n_greater = -1;
    
    #if DEBUG
    printArray(N,A);
    printf("partitioning at root - pivot = %ld, N = %d\n",pivot, N);
    #endif

    partition (pivot, N, A, &n_less, &n_equal, &n_greater);

    #if DEBUG
    printf("partitioned at root - less = %d, eq = %d, gt = %d\n",n_less, n_equal, n_greater);
    #endif

    assert (n_less >= 0 && n_equal >= 0 && n_greater >= 0);
    _Cilk_spawn quickSort (n_less, A);
    quickSort (n_greater, A + n_less + n_equal); 
//    _Cilk_sync;
  }
}

void
parallelSort (int N, keytype* A)
{
  quickSort (N, A);
}


/* eof */
