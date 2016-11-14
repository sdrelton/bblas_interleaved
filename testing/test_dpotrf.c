#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include <mkl.h>
#include <hbwmalloc.h>

#define nbtest 10
#define BATCH_COUNT 10000
#define MAX_BLOCK_SIZE 256
#define MAX_M 32
#define MAX_RHS 1
#define CACHECLEARSIZE 10000000
#define clearcache() cblas_ddot(CACHECLEARSIZE, bigA, 1, bigB, 1)


#define gettime() gettimeofday(&tv, NULL); time = tv.tv_sec*1000000+tv.tv_usec


int main(int arc, char *argv[])
{

  // Timer
  double time, time_intl, time_mkl;
  double timediff;
  double perf_mkl;
  struct timeval tv;
  // Info
  int nbconvtest;
    
  //Interleave variables
  double *arrayA = NULL;
  double norm;
  double error_blkintl;
  double perf_intl;
  int startpos;
  int ctr;

  //Block interleave variables
  double *work;
  double time_blkintl;
  double time_bestblkintl;
  double perf_blkintl;

  // Now create pointer-to-pointer batch of random matrices
  double **Ap2p =
    (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
  double **Aref =
    (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
  double **Asol =
    (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);

  // Setup parameters
  enum BBLAS_SIDE  side = BblasLeft;
  const int batch_count = BATCH_COUNT;
  int seed[4] = {2, 4, 1, 7}; // random seed
  int info = 0;
  int lda;
  double flops;

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

  printf("N,perf(LAPACKE +OMP), perf(blkintl+conv), bsize+conv,\
ratio(mkl/(blkintl+conv)), error(blkintl)\n");
  
  for (int N = 2; N < 33; N++){
      lda = N;
      flops = (1.0/3.*N*N*N + 1./2.*N*N + 1./6.*N)*BATCH_COUNT;
      
      // Generate batch
      for (int idx = 0; idx < BATCH_COUNT; idx++)
	{
	  // Generate A
	  Ap2p[idx] = (double*) hbw_malloc(sizeof(double) * N*N);
	  Asol[idx] = (double*) hbw_malloc(sizeof(double) * N*N);
	  Aref[idx] = (double*) hbw_malloc(sizeof(double) * N*N);
	  LAPACKE_dlarnv_work(IONE, ISEED, N*N, Ap2p[idx]);
	  for (int i = 0; i < N; i++)
	    Aref[idx][i*lda+i] +=N;
	}

      //=================================================
      // Compute result using LAPACKE
      //=================================================
      time_mkl =0.0;
      for (int testid = 0; testid < nbtest; testid++){
	memcpy_bptp2ptp(Ap2p, Aref, N, N, batch_count);
	clearcache();    
	gettime();
	timediff = time;
	#pragma omp parallel for
	for (int idx = 0; idx < batch_count; idx++) {
	    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', N, Ap2p[idx], lda);
	}
	gettime();
	timediff = time - timediff;
	if(testid != 0) time_mkl += timediff;
      }
      time_mkl /= (nbtest-1);
      perf_mkl = flops / time_mkl / 1000;
      
      //Copy the solution
      memcpy_bptp2ptp(Asol, Ap2p, N, N, batch_count);

      time_bestblkintl = 100000*time_mkl; //initialization
      double best_block = 0;
      for (int BLOCK_SIZE = 16; BLOCK_SIZE <= MAX_BLOCK_SIZE; BLOCK_SIZE +=16) {
	
	// Create block interleaved
	int blocksrequired = batch_count / BLOCK_SIZE;
	int remainder = 0;
	if (batch_count % BLOCK_SIZE != 0)
	  {
	    blocksrequired += 1;
	    remainder = batch_count % BLOCK_SIZE;
	  }
	work = (double*) hbw_malloc(sizeof(double) *N*N*BLOCK_SIZE*blocksrequired);
      
	//===========================================
	//Block interleave with internal conversion
	//==========================================
	double time_blkintl =0.0;
	for (int testid = 0; testid < nbtest; testid++){
	  memcpy_bptp2ptp(Ap2p, Aref, N, N, batch_count);
	  clearcache();    
	  gettime();
	  timediff = time;
	  bblas_dpotrf_batch_blkintl(CblasLower, N, Ap2p, lda,
				     BLOCK_SIZE, work, batch_count, info);
	  gettime();
	  timediff = time - timediff;
	  if(testid != 0) time_blkintl += timediff;
	}
	time_blkintl /= (nbtest-1);
	
	//Set best time and best block (with internal conversion)
	if ( time_blkintl < time_bestblkintl ) {
	  time_bestblkintl = time_blkintl;
	  best_block = BLOCK_SIZE;
	}
	hbw_free(work);
      }
      perf_blkintl = flops / time_bestblkintl / 1000;
      
      // Calculate difference between results
      double error_blkintl =  0.0;
      for (int idx = 0; idx < batch_count; idx++) {
	double error = 0.0;
	for (int i = 0; i < N; i++)
	  for (int j = 0; j <= i; j++ ) {
	    error += fabs(Ap2p[idx][i+lda*j] - Asol[idx][i+lda*j]);
	  }
	if (error_blkintl < error) error_blkintl = error;
      }

      printf("%d,%.2e,%.2e, %2.0f, %.2f,%.2e\n", N,
	     perf_mkl, perf_blkintl, best_block, perf_blkintl/perf_mkl, error_blkintl);   
      // Hbw_Free memory
      for (int idx = 0; idx < batch_count; idx++)
	{
	  hbw_free(Ap2p[idx]);
	  hbw_free(Aref[idx]);
	  hbw_free(Asol[idx]);
	}
  }

  hbw_free(Ap2p);
  hbw_free(Aref);
  hbw_free(Asol);
  free(bigA);
  free(bigB);
  return 0;
}

