#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>
#include <unistd.h>

#define BATCH_COUNT 1000
#define CACHECLEARSIZE 1000
#define clearcache() cblas_dgemm(colmaj, transA, transB, \
								 CACHECLEARSIZE, CACHECLEARSIZE, CACHECLEARSIZE,\
								 (alpha), bigA, CACHECLEARSIZE, \
                                 bigA, CACHECLEARSIZE, 		\
								 (beta),	\
								 bigC, CACHECLEARSIZE)


#define gettime() gettimeofday(&tv, NULL); time = tv.tv_sec*1000000+tv.tv_usec
#define TIMINGRUNS 10

int main()
{
// Timer
double time;
double timediff;
struct timeval tv;

for (int matsize = 2; matsize <= 10; matsize++)
{
int M = matsize;
int N = matsize;
int K = matsize;
double flops = 1.0 * (2*M*N*K) * BATCH_COUNT;
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

// Compute result using CBLAS
printf("Computing results using CBLAS (OpenMP)\n");
// Get prior time
float timings[TIMINGRUNS];
float avgtime;
for (int run = 0; run < TIMINGRUNS; run++)
{
	// Clear cache
	clearcache();
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
	timings[run] = timediff;
}
avgtime = 0;
for (int run = 0; run < TIMINGRUNS; run++)
{
	avgtime += timings[run];
}
avgtime /= TIMINGRUNS;
printf("CBLAS Time = %f us\n", avgtime);
printf("CBLAS Perf = %f GFlop/s\n\n", flops / avgtime / 1000);

// Compute result using interleaved
printf("Computing result using interleaved format\n");
for (int run = 0; run < TIMINGRUNS; run++)
{
	// Clear cache
	clearcache();
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
	timings[run] = timediff;
}
avgtime = 0;
for (int run = 0; run < TIMINGRUNS; run++)
{
	avgtime += timings[run];
}
printf("INTL Time = %f us\n", avgtime);
printf("INTL Perf = %f GFlop/s\n\n", flops / avgtime / 1000);

// Interleaved with OpenMP
printf("Computing result using interleaved format (OpenMP)\n");
memcpy(arrayC, arrayCorig, sizeof(double)*M*N*BATCH_COUNT);
for (int run = 0; run < TIMINGRUNS; run++)
{
	// Clear cache

	clearcache();
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
	timings[run] = timediff;
}
avgtime = 0;
for (int run = 0; run < TIMINGRUNS; run++)
{
	avgtime += timings[run];
}
printf("INTL_OPT Time = %f us\n", timediff);
printf("INTL_OPT Perf = %f GFlop/s\n\n", flops / timediff / 1000);

// Block Interleaved with OpenMP
printf("Computing result using block interleaved format (OpenMP)\n");
float besttime = 1e12;
int bestblock_size = 0;
for (int block_size = 2; block_size < batch_count; block_size *= 2)
{
// Create block interleaved
int blocksrequired = batch_count / block_size;
int remainder = 0;
if (batch_count % block_size != 0)
{
	blocksrequired += 1;
	remainder = batch_count % block_size;
}
double *arrayAblk = (double*)
	malloc(sizeof(double) * M*K*blocksrequired*block_size);
double *arrayBblk = (double*)
	malloc(sizeof(double) * K*N*blocksrequired*block_size);
double *arrayCblk = (double*)
	malloc(sizeof(double) * M*N*blocksrequired*block_size);
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
				arrayAblk[ctr] = Ap2p[blkidx * block_size + idx][pos];
				ctr++;
			}
			ctr += block_size - remainder;
		}
	}
	else
	{
		for (int pos = 0; pos < M*K; pos++)
		{
			for (int idx = 0; idx < block_size; idx++)
			{
				arrayAblk[ctr] = Ap2p[blkidx * block_size + idx][pos];
				ctr++;
			}
		}
	}
}

// Allocate B block interleaved
for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
{
	startpos = blkidx * block_size * K*N;
	if (blkidx == blocksrequired - 1 && remainder != 0)
	{
		// Remainders
		ctr = 0;
		for (int pos = 0; pos < K*N; pos++)
		{
			for (int idx = 0; idx < remainder; idx++)
			{
				arrayBblk[startpos + ctr] = Bp2p[blkidx * block_size + idx][pos];
				ctr++;
			}
			ctr += block_size - remainder;
		}
	}
	else
	{
		ctr = 0;
		for (int pos = 0; pos < K*N; pos++)
		{
			for (int idx = 0; idx < block_size; idx++)
			{
				arrayBblk[startpos + ctr] = Bp2p[blkidx * block_size + idx][pos];
				ctr++;
			}
		}
	}
}

// Allocate C block interleaved
for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
{
	startpos = blkidx * block_size * M*N;
	if (blkidx == blocksrequired - 1 && remainder != 0)
	{
		// Remainders
		ctr = 0;
		for (int pos = 0; pos < M*N; pos++)
		{
			for (int idx = 0; idx < remainder; idx++)
			{
				arrayCblk[startpos + ctr] = Cp2p[blkidx * block_size + idx][pos];
				ctr++;
			}
			ctr += block_size - remainder;
		}
	}
	else
	{
		ctr = 0;
		for (int pos = 0; pos < M*N; pos++)
		{
			for (int idx = 0; idx < block_size; idx++)
			{
				arrayCblk[startpos + ctr] = Cp2p[blkidx * block_size + idx][pos];
				ctr++;
			}
		}
	}
}
sleep(0.5);
for (int run = 0; run < TIMINGRUNS; run++)
{
	// Clear cache

	clearcache();
	gettime();
	timediff = time;

	bblas_dgemm_batch_blkintl(
		transA, transB,
		M, N, K,
		alpha,
		(const double*) arrayAblk,
		(const double*) arrayBblk,
		beta,
		arrayCblk, block_size,
		batch_count, info);
	gettime();
	timediff = time - timediff;
	timings[run] = timediff;
}
avgtime = 0;
for (int run = 0; run < TIMINGRUNS; run++)
{
	avgtime += timings[run];
}
if (avgtime < besttime)
{
	besttime = avgtime;
	bestblock_size = block_size;
}
free(arrayAblk);
free(arrayBblk);
free(arrayCblk);

}
printf("BLKINTL Time = %f us\n", besttime);
printf("BLKINTL Perf = %f GFlop/s\n", flops / besttime / 1000);
printf("BLKINTL Size = %d \n\n", bestblock_size);

// Free memory
free(arrayA);
free(arrayB);
free(arrayC);
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

printf("\n\n\n");

}
return 0;
}
