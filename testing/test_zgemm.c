//#include <cblas.h>
//#include <lapacke.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>

#define M 8
#define N 8
#define K 8
#define BATCH_COUNT 50000

#define CACHECLEARSIZE = 5000

#define gettime() gettimeofday(&tv, NULL); time = tv.tv_sec*1000000+tv.tv_usec

int main()
{
// Timer
double time;
double timediff;
double flops = 1.0 * (2*M*N*K) * BATCH_COUNT;
struct timeval tv;

// Info
printf("M = %d\n", M);
printf("N = %d\n", N);
printf("K = %d\n", K);
printf("batch_count = %d\n", BATCH_COUNT);
#pragma omp parallel
{
	if (omp_get_thread_num() == 0)
	{
		printf("OMP threads = %d\n", omp_get_num_threads());
	}
}
printf("MKL threads = %d\n", mkl_get_max_threads());
printf("GFlops = %f\n", flops/1e9);

// Generate batch
int seed[4] = {2, 4, 1, 7}; // random seed
int colmaj = BblasColMajor; // Use column major ordering

// Needed to generate random matrices using LAPACKE_zlagge
const int len = max(M, max(N, K));
double *scalar = (double*) malloc (sizeof(double) * len);
double val;
val = 1.0;
for (int i = 0; i < len; i++)
{
	scalar[i] = val;
	val += 1;
}

// Now create pointer-to-pointer batch of random matrices
printf("Generating random matrices\n");
BBLAS_Complex64_t **Ap2p =
	(BBLAS_Complex64_t**) malloc(sizeof(BBLAS_Complex64_t*)*BATCH_COUNT);
BBLAS_Complex64_t **Bp2p =
	(BBLAS_Complex64_t**) malloc(sizeof(BBLAS_Complex64_t*)*BATCH_COUNT);
BBLAS_Complex64_t **Cp2p =
	(BBLAS_Complex64_t**) malloc(sizeof(BBLAS_Complex64_t*)*BATCH_COUNT);

for (int idx = 0; idx < BATCH_COUNT; idx++)
{
	// Generate A
	Ap2p[idx] = (BBLAS_Complex64_t*) malloc(sizeof(BBLAS_Complex64_t) * M*K);
	LAPACKE_zlagge(colmaj, M, K, M-1, K-1, scalar, Ap2p[idx], M, seed);

	// Generate B
	Bp2p[idx] = (BBLAS_Complex64_t*) malloc(sizeof(BBLAS_Complex64_t) * K*N);
	LAPACKE_zlagge(colmaj, K, N, K-1, N-1, scalar, Bp2p[idx], K, seed);
	// Generate C
	Cp2p[idx] = (BBLAS_Complex64_t*) malloc(sizeof(BBLAS_Complex64_t) * M*N);
	LAPACKE_zlagge(colmaj, M, N, M-1, N-1, scalar, Cp2p[idx], M, seed);
}
free(scalar);
//free(seed);

// Setup parameters
const enum BBLAS_TRANS transA = BblasNoTrans;
const enum BBLAS_TRANS transB = BblasNoTrans;
const BBLAS_Complex64_t alpha = 1.0;
const BBLAS_Complex64_t beta = 0.0;
const int lda = M;
const int ldb = K;
const int ldc = M;
const int batch_count = BATCH_COUNT;
const int strideA = BATCH_COUNT;
const int strideB = BATCH_COUNT;
const int strideC = BATCH_COUNT;
int info = 0;

// Create interleaved matrices
printf("Converting to interleaved format\n\n");
BBLAS_Complex64_t *arrayA = (BBLAS_Complex64_t*)
	malloc(sizeof(BBLAS_Complex64_t) * lda*K*batch_count);
BBLAS_Complex64_t *arrayB = (BBLAS_Complex64_t*)
	malloc(sizeof(BBLAS_Complex64_t) * ldb*N*batch_count);
BBLAS_Complex64_t *arrayC = (BBLAS_Complex64_t*)
	malloc(sizeof(BBLAS_Complex64_t) * ldc*N*batch_count);
BBLAS_Complex64_t *arrayCorig = (BBLAS_Complex64_t*)
	malloc(sizeof(BBLAS_Complex64_t) * ldc*N*batch_count);
int ctr;

// Allocate A interleaved
ctr = 0;
for (int pos = 0; pos < M*K; pos++)
{
	for (int idx = 0; idx < batch_count; idx++)
	{
		arrayA[ctr] = Ap2p[idx][pos];
		ctr++;
	}
}

// Allocate B interleaved
ctr = 0;
for (int pos = 0; pos < K*N; pos++)
{
	for (int idx = 0; idx < batch_count; idx++)
	{
		arrayB[ctr] = Bp2p[idx][pos];
		ctr++;
	}
}

// Allocate C interleaved
ctr = 0;
for (int pos = 0; pos < M*N; pos++)
{
	for (int idx = 0; idx < batch_count; idx++)
	{
		arrayC[ctr] = Cp2p[idx][pos];
		ctr++;
	}
}

memcpy(arrayCorig, arrayC, sizeof(BBLAS_Complex64_t)*M*N*BATCH_COUNT);

// Compute result using CBLAS
printf("Computing results using CBLAS (OpenMP)\n");
// Get prior time
gettime();
timediff = time;
#pragma omp parallel for
for (int idx = 0; idx < batch_count; idx++)
{
	cblas_zgemm(
		BblasColMajor,
		transA,
		transB,
		M,
		N,
		K,
		CBLAS_SADDR(alpha),
		Ap2p[idx],
		lda,
		Bp2p[idx],
		ldb,
		CBLAS_SADDR(beta),
		Cp2p[idx],
		ldc);
}
gettime();
timediff = time - timediff;
printf("CBLAS Time = %f us\n", timediff);
printf("CBLAS Perf = %f GFlop/s\n\n", flops / timediff / 1000);

// Compute result using interleaved
printf("Computing result using interleaved format\n");
// Get prior time
gettime();
timediff = time;
bblas_zgemm_batch_intl(
	transA, transB,
	M, N, K,
	alpha,
	(const BBLAS_Complex64_t*) arrayA, strideA,
	(const BBLAS_Complex64_t*) arrayB, strideB,
	beta,
	arrayC, strideC,
	batch_count, info);
gettime();
timediff = time - timediff;
printf("INTL Time = %f us\n", timediff);
printf("INTL Perf = %f GFlop/s\n\n", flops / timediff / 1000);

// Interleaved with OpenMP
printf("Computing result using interleaved format (OpenMP)\n");
memcpy(arrayC, arrayCorig, sizeof(BBLAS_Complex64_t)*M*N*BATCH_COUNT);
// Get prior time
gettime();
timediff = time;
bblas_zgemm_batch_intl_opt(
	transA, transB,
	M, N, K,
	alpha,
	(const BBLAS_Complex64_t*) arrayA, strideA,
	(const BBLAS_Complex64_t*) arrayB, strideB,
	beta,
	arrayC, strideC,
	batch_count, info);
gettime();
timediff = time - timediff;
printf("INTL_OPT Time = %f us\n", timediff);
printf("INTL_OPT Perf = %f GFlop/s\n\n", flops / timediff / 1000);

// Calculate difference between results
printf("Calculating l1 difference between results\n");
double norm = 0;
ctr = 0;
for (int j = 0; j < N; j++)
{
	for (int i = 0; i < M; i++)
	{
		for (int idx = 0; idx < batch_count; idx++)
		{
			norm += cabs(arrayC[ctr] - Cp2p[idx][j*ldc+i]);
			ctr++;
		}
	}
}
printf("norm = %f\n", norm);

// Free memory
free(arrayA);
free(arrayB);
free(arrayC);
for (int idx = 0; idx < batch_count; idx++)
{
	free(Ap2p[idx]);
	free(Bp2p[idx]);
	free(Cp2p[idx]);
}
free(Ap2p);
free(Bp2p);
free(Cp2p);


return 0;
}
