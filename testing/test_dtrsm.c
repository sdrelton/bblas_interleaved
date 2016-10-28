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
#define BLOCK_SIZE 128
#define CACHECLEARSIZE 10000
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
    double flops = 1.0 * (N*M*M)*BATCH_COUNT;
    struct timeval tv;

    int ISEED[4] ={0,0,0,1};
    int IONE = 1;
    // Info
    printf("M = %d\n", M);
    printf("N = %d\n", N);
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
    
    printf("Generating random matrices to clear cache\n");
    // Generate matrices to clear cache
    int bigsize = CACHECLEARSIZE;
    double* bigA =
        (double*) malloc(sizeof(double) * bigsize*bigsize);
    double* bigB =
        (double*) malloc(sizeof(double) * bigsize*bigsize);
    double* bigC =
        (double*) malloc(sizeof(double) * bigsize*bigsize);
    
    LAPACKE_dlarnv_work(IONE, ISEED, bigsize*bigsize, bigA);
    LAPACKE_dlarnv_work(IONE, ISEED, bigsize*bigsize, bigB);
    LAPACKE_dlarnv_work(IONE, ISEED, bigsize*bigsize, bigC);

    
    printf("Generating random matrices for computation\n");
    // Now create pointer-to-pointer batch of random matrices
    double **Ap2p =
        (double**) malloc(sizeof(double*)*BATCH_COUNT);
    double **Bp2p =
        (double**) malloc(sizeof(double*)*BATCH_COUNT);

    for (int idx = 0; idx < BATCH_COUNT; idx++)
    {
        // Generate A
        Ap2p[idx] = (double*) malloc(sizeof(double) * M*M);
        LAPACKE_dlarnv_work(IONE, ISEED, M*M, Ap2p[idx]);
        for (int i = 0; i < M; i++)
            Ap2p[idx][i*M+i] +=1;

        // Generate B
        Bp2p[idx] = (double*) malloc(sizeof(double) * M*N);
        LAPACKE_dlarnv_work(IONE, ISEED, M*N, Bp2p[idx]);
    }

//free(seed);
    
    // Setup parameters
    enum BBLAS_SIDE  side = BblasLeft;
    enum BBLAS_UPLO uplo  = BblasLower;
    enum BBLAS_DIAG diag  = BblasNonUnit;
    enum BBLAS_TRANS transA = BblasNoTrans;
    enum BBLAS_TRANS transB = BblasNoTrans;
    const double alpha = 1.0;
    const double beta = 0.0;
    const int lda = M;
    const int ldb = M;
    const int batch_count = BATCH_COUNT;
    const int strideA = BATCH_COUNT;
    const int strideB = BATCH_COUNT;
    int info = 0;

    // Create interleaved matrices
    printf("Converting to interleaved format\n\n");
    double *arrayA = (double*)
        malloc(sizeof(double) * lda*M*batch_count);
    double *arrayB = (double*)
        malloc(sizeof(double) * ldb*N*batch_count);
    double *arrayBref = (double*)
        malloc(sizeof(double) * ldb*N*batch_count);
    int ctr;
    
    // Allocate A interleaved
    ctr = 0;
    for (int j = 0; j < M; j++) {
        for (int i =  0; i < M; i++ ) {
            for (int idx = 0; idx < batch_count; idx++)
            {
                arrayA[ctr] = Ap2p[idx][j*lda+i];
                ctr++;
            }
        }
    }

    // Allocate B interleaved
    ctr = 0;
    for (int pos = 0; pos < M*N; pos++)
    {
        for (int idx = 0; idx < batch_count; idx++)
        {
            arrayB[ctr] = Bp2p[idx][pos];
            ctr++;
        }
    }
    
    memcpy(arrayBref, arrayB, sizeof(double)*M*N*BATCH_COUNT);
    
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
        cblas_dtrsm(
            BblasColMajor, side, uplo, transA, diag,
            M, N, alpha, Ap2p[idx], lda, Bp2p[idx], ldb);
    }
    gettime();
    timediff = time - timediff;
    printf("CBLAS Time = %f us\n", timediff);
    printf("CBLAS Perf = %f GFlop/s\n\n", flops / timediff / 1000);
    
    // Clear cache
    printf("Clearing cache\n");
    clearcache();
    
    // Interleaved with OpenMP
    printf("Computing result using interleaved format (OpenMP)\n");
    memcpy(arrayB, arrayBref, sizeof(double)*M*N*BATCH_COUNT);
    
    // Get prior time
    gettime();
    timediff = time;
    bblas_dtrsm_batch_intl(
        side, uplo, transA, diag,
        M, N, alpha, arrayA, strideA,
        (const double*) arrayB, strideB,
        batch_count, info);
    gettime();
    timediff = time - timediff;
    printf("INTL Time = %f us\n", timediff);
    printf("INTL Perf = %f GFlop/s\n\n", flops / timediff / 1000);
    

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
			norm += cabs(arrayB[ctr] - Bp2p[idx][j*lda+i]);
			ctr++;
		}
	}
}
printf("INTL norm = %f\n", norm);


// Free memory
free(arrayA);
free(arrayB);
free(arrayBref);
for (int idx = 0; idx < batch_count; idx++)
{
	free(Ap2p[idx]);
	free(Bp2p[idx]);
}
free(Ap2p);
free(Bp2p);


return 0;
}
