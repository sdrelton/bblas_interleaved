//#include <cblas.h>
//#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>

#define M 2
#define N 2
#define K 2
#define BATCH_COUNT 100000

#define CACHECLEARSIZE 1000
#define clearcache() cblas_dgemm(colmaj, transA, transB, \
								 CACHECLEARSIZE, CACHECLEARSIZE, CACHECLEARSIZE,\
								 (alpha), bigA, CACHECLEARSIZE, \
                                 bigA, CACHECLEARSIZE, 		\
								 (beta),	\
								 bigC, CACHECLEARSIZE)


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

// Needed to generate random matrices using LAPACKE_dlagge
const int len = max(M, max(N, max(CACHECLEARSIZE, K)));
double *scalar = (double*) malloc (sizeof(double) * len);
double val;
val = 1.0;
for (int i = 0; i < len; i++)
{
	scalar[i] = val;
	val += 1;
}

printf("Generating random matrices to clear cache\n");
// Generate matrices to clear cache
int bigsize = CACHECLEARSIZE;
double* bigA =
	(double*) malloc(sizeof(double) * bigsize*bigsize);
double* bigC =
	(double*) malloc(sizeof(double) * bigsize*bigsize);
LAPACKE_dlagge(colmaj, bigsize, bigsize, bigsize-1, bigsize-1, scalar, bigA, bigsize, seed);
LAPACKE_dlagge(colmaj, bigsize, bigsize, bigsize-1, bigsize-1, scalar, bigC, bigsize, seed);

printf("Generating random matrices for computation\n");
// Now create pointer-to-pointer batch of random matrices
double **Ap2p =
	(double**) malloc(sizeof(double*)*BATCH_COUNT);
double **Bp2p =
	(double**) malloc(sizeof(double*)*BATCH_COUNT);
double **Cp2p =
	(double**) malloc(sizeof(double*)*BATCH_COUNT);

for (int idx = 0; idx < BATCH_COUNT; idx++)
{
	// Generate A
	Ap2p[idx] = (double*) malloc(sizeof(double) * M*K);
	LAPACKE_dlagge(colmaj, M, K, M-1, K-1, scalar, Ap2p[idx], M, seed);

	// Generate B
	Bp2p[idx] = (double*) malloc(sizeof(double) * K*N);
	LAPACKE_dlagge(colmaj, K, N, K-1, N-1, scalar, Bp2p[idx], K, seed);
	// Generate C
	Cp2p[idx] = (double*) malloc(sizeof(double) * M*N);
	LAPACKE_dlagge(colmaj, M, N, M-1, N-1, scalar, Cp2p[idx], M, seed);
}
free(scalar);
//free(seed);

// Setup parameters
const enum BBLAS_TRANS transA = BblasNoTrans;
const enum BBLAS_TRANS transB = BblasNoTrans;
const double alpha = 1.0;
const double beta = 0.0;
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
double *arrayA = (double*)
	malloc(sizeof(double) * lda*K*batch_count);
double *arrayB = (double*)
	malloc(sizeof(double) * ldb*N*batch_count);
double *arrayC = (double*)
	malloc(sizeof(double) * ldc*N*batch_count);
double *arrayCorig = (double*)
	malloc(sizeof(double) * ldc*N*batch_count);
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

memcpy(arrayCorig, arrayC, sizeof(double)*M*N*BATCH_COUNT);

// Clear cache
printf("Clearing cache\n");
clearcache();

// Compute result using CBLAS
printf("Computing results using CBLAS (OpenMP)\n");
// Get prior time
gettime();
timediff = time;
#pragma omp parallel for
for (int idx = 0; idx < batch_count; idx++)
{
	cblas_dgemm(
		BblasColMajor,
		transA,
		transB,
		M,
		N,
		K,
		(alpha),
		Ap2p[idx],
		lda,
		Bp2p[idx],
		ldb,
		(beta),
		Cp2p[idx],
		ldc);
}
gettime();
timediff = time - timediff;
printf("CBLAS Time = %f us\n", timediff);
printf("CBLAS Perf = %f GFlop/s\n\n", flops / timediff / 1000);

// Clear cache
printf("Clearing cache\n");
clearcache();

// Compute result using interleaved
printf("Computing result using interleaved format\n");
// Get prior time
gettime();
timediff = time;
bblas_dgemm_batch_intl(
	transA, transB,
	M, N, K,
	alpha,
	(const double*) arrayA, strideA,
	(const double*) arrayB, strideB,
	beta,
	arrayC, strideC,
	batch_count, info);
gettime();
timediff = time - timediff;
printf("INTL Time = %f us\n", timediff);
printf("INTL Perf = %f GFlop/s\n\n", flops / timediff / 1000);

// Clear cache
printf("Clearing cache\n");
clearcache();

// Interleaved with OpenMP
printf("Computing result using interleaved format (OpenMP)\n");
memcpy(arrayC, arrayCorig, sizeof(double)*M*N*BATCH_COUNT);
// Get prior time
gettime();
timediff = time;
bblas_dgemm_batch_intl_opt(
	transA, transB,
	M, N, K,
	alpha,
	(const double*) arrayA, strideA,
	(const double*) arrayB, strideB,
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
