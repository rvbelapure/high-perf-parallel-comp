/**
 *  \file parallel-mergesort--omp.cc
 *
 *  \brief Implement your parallel mergesort in this file.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "sort.hh"
#include <algorithm>

#include <omp.h>

#define DEBUG 0

void parallelSort(int N, keytype* A);
void p_mergesort(keytype * A, int p, int r, keytype * B, int s);
void parallel_merge(keytype * T, int p1, int r1, int p2, int r2, keytype * A, int p3);
int binary_search(keytype x, keytype * T, int p, int r);
void printArray(int N, keytype * A);


void printArray(int N, keytype * A, char * tag)
{
	#if DEBUG
	printf("%s: ", tag);
	for(int i = 0 ; i < N ; i++)
		printf("%ld ",A[i]);
	printf("\n");
	#endif
}


int binary_search(keytype x, keytype * T, int p, int r)
{
	int low = p;
	int high = (p > (r + 1)) ? p : (r+1);
	while(low < high)
	{
		int mid = (low + high) / 2;
		if(x <= T[mid])
			high = mid;
		else
			low = mid + 1;
	}
	return high;
}

void parallel_merge(keytype *T, int p1, int r1, int p2, int r2, keytype *A, int p3)
{
	int n1 = r1 - p1 + 1;
	int n2 = r2 - p2 + 1;
	if(n1 < n2)
	{
		std::swap(p1, p2);
		std::swap(r1, r2);
		std::swap(n1, n2);
	}
	if(n1 == 0)
		return;
	int q1 = (p1 + r1) / 2;
	int q2 = binary_search(T[q1], T, p2, r2);
	int q3 = p3 + (q1 - p1) + (q2 - p2);
	A[q3] = T[q1];
	#pragma omp task default(none) shared(T, A, p1, q1, p2, q2, p3)
	parallel_merge(T, p1, q1 - 1, p2, q2 - 1, A, p3);
	#pragma omp task default(none) shared(T, A, q1, r1, q2, r2, q3)
	parallel_merge(T, q1 + 1, r1, q2, r2, A, q3 + 1);
	#pragma omp taskwait
}

void p_mergesort(keytype * A, int p, int r, keytype * B, int s)
{
	int G = 500;
	int n = r - p + 1;
	if(n < G)
	{
		sequentialSort(n, B + s);		// Assume, B will always be copy of A when passed to p_mergesort
		return;
	}

	keytype * T = newCopy(r+1, A);
	printArray(r+1, T, "A");

	int q = ((p + r) / 2);
	#pragma omp task default(none) shared(A, T, p, q)
	p_mergesort(A, p, q, T, p);
	#pragma omp task default(none) shared(A, T, q, r)
	p_mergesort(A, q + 1, r, T, q + 1);
	#pragma omp taskwait
	parallel_merge(T, p, q, q + 1, r, B, s);

	printArray(n, B + s,"merged");
	free(T);
}


void
parallelSort (int N, keytype* A)
{
	keytype * B = newCopy(N, A);
	#pragma omp parallel
	#pragma omp single
	p_mergesort(A, 0, N-1, B, 0);
	int i;
	#pragma omp parallel for default(none) shared(A,B,N) private(i)
	for(i = 0 ; i < N ; i++)
		A[i] = B[i];
	free(B);
	printArray(N, A, "final : ");
}

/* eof */
