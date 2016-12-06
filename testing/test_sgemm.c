//#include <cblas.h>
//#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <hbwmalloc.h>
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>

#define nbtest 10
#define BATCH_COUNT 10240
#define MAX_BLOCK_SIZE 256
#define MAX_M 32
#define CACHECLEARSIZE 10000000
#define clearcache() cblas_ddot(CACHECLEARSIZE, bigA, 1, bigB, 1)


#define gettime() gettimeofday(&tv, NULL); time = tv.tv_sec*1000000+tv.tv_usec

int main(int arc, char *argv[]) {
  
  // Timer
  double time, time_intl, time_mkl;
  double timediff;
  double perf_mkl;
  struct timeval tv;

  // Generate matrices to clear cach
  int ISEED[4] ={0,0,0,1};
  int IONE = 1;
  int bigsize = CACHECLEARSIZE;
  double* bigA =
    (double*) malloc(sizeof(double) * bigsize);
  double* bigB =
    (double*) malloc(sizeof(double) *bigsize);
  LAPACKE_dlarnv_work(IONE, ISEED, bigsize, bigA);
  LAPACKE_dlarnv_work(IONE, ISEED, bigsize, bigB);
  
  printf("Generating random matrices for computation\n");
  // Now create pointer-to-pointer batch of random matrices
  float **Ap2p =
    (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
  float **Bp2p =
    (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
  float **Cp2p =
    (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
  float **Cref =
    (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
  float **Csol =
    (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);


  // Setup parameters
  float *work;
  enum BBLAS_TRANS transA = BblasNoTrans;
  enum BBLAS_TRANS transB = BblasNoTrans;
  float alpha = 1.0;
  float beta = 1.0;
  int batch_count = BATCH_COUNT;
  int info = 0;

  printf("M,K,N, perf_cblas, perf_batch_mkl, perf_blkintl, blocksize,\
 ratio(batch_mkl/blkintl), error(batch_mkl), error(blk)\n");

  for (int M = 2; M < MAX_M; M++) {
    int K = M;
    int N = M;
    int lda = M;
    int ldb = M;
    int ldc = M;
    double flops = 1.0 * (2*M*N*K) * BATCH_COUNT;

    // Generate batch
    for (int idx = 0; idx < BATCH_COUNT; idx++) {
      // Generate A
      Ap2p[idx] = (float*) hbw_malloc(sizeof(float) * lda*K);
      LAPACKE_slarnv_work(IONE, ISEED, lda*K, Ap2p[idx]);
      // Generate B
      Bp2p[idx] = (float*) hbw_malloc(sizeof(float) * ldb*N);
      LAPACKE_slarnv_work(IONE, ISEED, ldb*N, Bp2p[idx]);
      // Generate C
      Cp2p[idx] = (float*) hbw_malloc(sizeof(float) * ldc*N);
      Csol[idx] = (float*) hbw_malloc(sizeof(float) * ldc*N);
      Cref[idx] = (float*) hbw_malloc(sizeof(float) * ldc*N);
      LAPACKE_slarnv_work(IONE, ISEED, ldc*N, Cref[idx]);
    }    
    
    //=================================================
    // Compute result using CBLAS + OMP
    //=================================================
    double time_cblas =0.0;
    for (int testid = 0; testid < nbtest; testid++) {
      memcpy_sbptp2ptp(Cp2p, Cref, ldc, N, batch_count);
      clearcache();    
      gettime();
      timediff = time;
      #pragma omp parallel for
      for (int idx = 0; idx < batch_count; idx++) {
      	cblas_sgemm(CblasColMajor, transA, transB,
      		    M, N, K, alpha, Ap2p[idx], lda,
      		    Bp2p[idx], ldb, beta, Cp2p[idx], ldc);
      }
      gettime();
      timediff = time - timediff;
      if(testid != 0) time_cblas += timediff;
    }
    time_cblas /= (nbtest-1);
    double perf_cblas = flops / time_cblas / 1000;
    //Copy the solution
    memcpy_sbptp2ptp(Csol, Cp2p, ldc, N, batch_count);
    
    //=================================================
    // Compute result using MKL Batch
    //=================================================
    double time_mkl =0.0;
    for (int testid = 0; testid < nbtest; testid++) {
      memcpy_sbptp2ptp(Cp2p, Cref, ldc, N, batch_count);
      clearcache();    
      gettime();
      timediff = time;
      cblas_sgemm_batch(CblasColMajor, &transA, &transB,
      			&M, &N, &K, &alpha, Ap2p, &lda,
      			Bp2p, &ldb, &beta, Cp2p, &ldc,
      			1, &batch_count);
      
      gettime();
      timediff = time - timediff;
      if(testid != 0) time_mkl += timediff;
    }
    time_mkl /= (nbtest-1);
    perf_mkl = flops / time_mkl / 1000;
    float error_mkl =  get_serror(Csol, Cp2p, ldc, N, batch_count);    
    //===========================================
    //Block interleave with internal conversion
    //==========================================
    double time_bestblkintl = 100000*time_mkl; //initialization
    int best_block = 0;
    float error_blkintl;
    for (int BLOCK_SIZE = 8; BLOCK_SIZE <= MAX_BLOCK_SIZE; BLOCK_SIZE +=8) {
      // Create block interleaved
      int blocksrequired = batch_count / BLOCK_SIZE;
      int remainder = 0;
      if (batch_count % BLOCK_SIZE != 0)
	{
	  blocksrequired += 1;
	  remainder = batch_count % BLOCK_SIZE;
	}
      int lwork = (M*K + K*N + M*N)*BLOCK_SIZE*blocksrequired;
      work = (float*) hbw_malloc(sizeof(float)*lwork);
      
      double time_blkintl =0.0;
      for (int testid = 0; testid < nbtest; testid++){
	memcpy_sbptp2ptp(Cp2p, Cref, ldc, N, batch_count);
	clearcache();    
	gettime();
	timediff = time;
	bblas_sgemm_blkintl(transA, transB,
			    M, K, N, alpha, Ap2p,
			    Bp2p, beta,	Cp2p, work,
			    BLOCK_SIZE,	batch_count, info);
	gettime();
	timediff = time - timediff;
	if(testid != 0) time_blkintl += timediff;
	// Calculate difference between results
	if(testid != 0) error_blkintl =  get_serror(Csol, Cp2p, ldc, N, batch_count);  
      }
      time_blkintl /= (nbtest-1);
      if ( time_blkintl < time_bestblkintl ) {
	time_bestblkintl = time_blkintl;
	best_block = BLOCK_SIZE;
      }
      hbw_free(work);
    }
    double perf_blkintl = flops / time_bestblkintl / 1000;
    
    printf("%d, %d, %d, %.2e, %.2e, %.2e, %d, %.2f, %.2e, %.2e\n",
	   M, K, N, perf_cblas, perf_mkl, perf_blkintl,
	   best_block, perf_blkintl/perf_mkl, error_mkl, error_blkintl);
    
    for (int idx = 0; idx < batch_count; idx++) {
      hbw_free(Ap2p[idx]);
      hbw_free(Bp2p[idx]);
      hbw_free(Cp2p[idx]);
      hbw_free(Csol[idx]);
      hbw_free(Cref[idx]);
    }
  }
  hbw_free(Ap2p);
  hbw_free(Bp2p);
  hbw_free(Cp2p);
  hbw_free(Csol);
  hbw_free(Cref);
  free(bigA);
  free(bigB);
  return 0;
}
