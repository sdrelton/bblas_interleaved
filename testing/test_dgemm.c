//#include <cblas.h>
//#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>

#define M 3
#define N 3
#define K 3
#define BATCH_COUNT 10000
#define BLOCK_SIZE 512

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

// Create block interleaved
printf("Converting to block interleaved format - block_size = %d \n\n", BLOCK_SIZE);
int blocksrequired = batch_count / BLOCK_SIZE;
int remainder = 0;
if (batch_count % BLOCK_SIZE != 0)
{
	blocksrequired += 1;
	remainder = batch_count % BLOCK_SIZE;
}
double *arrayAblk = (double*)
	malloc(sizeof(double) * M*K*blocksrequired*BLOCK_SIZE);
double *arrayBblk = (double*)
	malloc(sizeof(double) * K*N*blocksrequired*BLOCK_SIZE);
double *arrayCblk = (double*)
	malloc(sizeof(double) * M*N*blocksrequired*BLOCK_SIZE);
int startpos;

// Allocate A block interleaved
ctr = 0;
for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
{
	if ((blkidx == blocksrequired - 1) && (remainder != 0))
	{
		// Remainders
		for (int pos = 0; pos < M*K; pos++)
		{
			for (int idx = 0; idx < remainder; idx++)
			{
				arrayAblk[ctr] = Ap2p[blkidx * BLOCK_SIZE + idx][pos];
				ctr++;
			}
			ctr += BLOCK_SIZE - remainder;
		}
	}
	else
	{
		for (int pos = 0; pos < M*K; pos++)
		{
			for (int idx = 0; idx < BLOCK_SIZE; idx++)
			{
				arrayAblk[ctr] = Ap2p[blkidx * BLOCK_SIZE + idx][pos];
				ctr++;
			}
		}
	}
}

// Allocate B block interleaved
for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
{
	startpos = blkidx * BLOCK_SIZE * K*N;
	if (blkidx == blocksrequired - 1 && remainder != 0)
	{
		// Remainders
		ctr = 0;
		for (int pos = 0; pos < K*N; pos++)
		{
			for (int idx = 0; idx < remainder; idx++)
			{
				arrayBblk[startpos + ctr] = Bp2p[blkidx * BLOCK_SIZE + idx][pos];
				ctr++;
			}
			ctr += BLOCK_SIZE - remainder;
		}
	}
	else
	{
		ctr = 0;
		for (int pos = 0; pos < K*N; pos++)
		{
			for (int idx = 0; idx < BLOCK_SIZE; idx++)
			{
				arrayBblk[startpos + ctr] = Bp2p[blkidx * BLOCK_SIZE + idx][pos];
				ctr++;
			}
		}
	}
}

// Allocate C block interleaved
for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
{
	startpos = blkidx * BLOCK_SIZE * M*N;
	if (blkidx == blocksrequired - 1 && remainder != 0)
	{
		// Remainders
		ctr = 0;
		for (int pos = 0; pos < M*N; pos++)
		{
			for (int idx = 0; idx < remainder; idx++)
			{
				arrayCblk[startpos + ctr] = Cp2p[blkidx * BLOCK_SIZE + idx][pos];
				ctr++;
			}
			ctr += BLOCK_SIZE - remainder;
		}
	}
	else
	{
		ctr = 0;
		for (int pos = 0; pos < M*N; pos++)
		{
			for (int idx = 0; idx < BLOCK_SIZE; idx++)
			{
				arrayCblk[startpos + ctr] = Cp2p[blkidx * BLOCK_SIZE + idx][pos];
				ctr++;
			}
		}
	}
}

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

// Clear cache
printf("Clearing cache\n");
clearcache();

// Block Interleaved with OpenMP
printf("Computing result using block interleaved format (OpenMP)\n");
// Get prior time
gettime();
timediff = time;
bblas_dgemm_batch_blkintl(
	transA, transB,
	M, N, K,
	alpha,
	(const double*) arrayAblk,
	(const double*) arrayBblk,
	beta,
	arrayCblk, BLOCK_SIZE,
	batch_count, info);
gettime();
timediff = time - timediff;
printf("BLKINTL Time = %f us\n", timediff);
printf("BLKINTL Perf = %f GFlop/s\n\n", flops / timediff / 1000);

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
printf("INTL norm = %f\n", norm);

norm = 0;
for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
{
	startpos = blkidx * BLOCK_SIZE * M*N;
	if (blkidx == blocksrequired - 1 && remainder != 0)
	{
		// Remainders
		ctr = 0;
		for (int pos = 0; pos < M*N; pos++)
		{
			for (int idx = 0; idx < remainder; idx++)
			{
				norm += cabs(arrayCblk[startpos + ctr] - Cp2p[blkidx * BLOCK_SIZE + idx][pos]);
				ctr++;
			}
			ctr += BLOCK_SIZE - remainder;
		}
	}
	else
	{
		ctr = 0;
		for (int pos = 0; pos < M*N; pos++)
		{
			for (int idx = 0; idx < BLOCK_SIZE; idx++)
			{
				norm += cabs(arrayCblk[startpos + ctr] - Cp2p[blkidx * BLOCK_SIZE + idx][pos]);
				ctr++;
			}
		}
	}
}
printf("BLOCK INTL norm = %f\n", norm);

// Free memory
free(arrayA);
free(arrayB);
free(arrayC);
free(arrayAblk);
free(arrayBblk);
free(arrayCblk);
free(arrayCorig);
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
