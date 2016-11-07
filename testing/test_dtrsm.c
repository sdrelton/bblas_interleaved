#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>
#include <hbwmalloc.h>

#define nbtest 5
#define BATCH_COUNT 10000
#define MAX_BLOCK_SIZE 256
#define MAX_M 32
#define CACHECLEARSIZE 10000
#define clearcache() cblas_dgemm(colmaj, transA, transB,		\
				 CACHECLEARSIZE, CACHECLEARSIZE, CACHECLEARSIZE, \
				 (alpha), bigA, CACHECLEARSIZE,		\
                                 bigA, CACHECLEARSIZE,			\
				 (beta),				\
				 bigC, CACHECLEARSIZE)


#define gettime() gettimeofday(&tv, NULL); time = tv.tv_sec*1000000+tv.tv_usec


int main(int arc, char *argv[])
{

  // Timer
  double time, time_intl, time_mkl;
  double timediff;
  double perf_mkl;
  struct timeval tv;
  int ISEED[4] ={0,0,0,1};
  int IONE = 1;
  // Info
  int nbconvtest;
    
  if (1) { 
    nbconvtest = nbtest;
  } else {
    nbconvtest = 1;
  }

  printf("Generating random matrices to clear cache\n");
  // Generate matrices to clear cache
  int bigsize = CACHECLEARSIZE;
  double* bigA =
    (double*) hbw_malloc(sizeof(double) * bigsize*bigsize);
  double* bigB =
    (double*) hbw_malloc(sizeof(double) * bigsize*bigsize);
  double* bigC =
    (double*) hbw_malloc(sizeof(double) * bigsize*bigsize);
    
  LAPACKE_dlarnv_work(IONE, ISEED, bigsize*bigsize, bigA);
  LAPACKE_dlarnv_work(IONE, ISEED, bigsize*bigsize, bigB);
  LAPACKE_dlarnv_work(IONE, ISEED, bigsize*bigsize, bigC);
    

  //Interleave variables
  double *arrayA = NULL;
  double *arrayB = NULL;
  double time_cpyint;
  double time_cpyB;
  double norm;
  double error_intl;
  double error_blkintl;
  double perf_intl;
  double perf_intl_conv;
  int startpos;
  int ctr;

  //Block interleave variables
  double *arrayAblk;
  double *arrayBblk;
  double time_cpyblkint;  
  double time_blkintl;
  double time_bestblkintl;
  double time_bestblkintl_conv;
  double perf_blkintl;
  double perf_blkintl_conv;
  int best_block;
  int best_block_conv;
  printf("Generating random matrices for computation\n");
  // Now create pointer-to-pointer batch of random matrices
  double **Ap2p =
    (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
  double **Bp2p =
    (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
  double **Bref =
    (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);

  double **Bsol =
    (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);


  // Setup parameters
  enum BBLAS_SIDE  side = BblasLeft;
  enum BBLAS_UPLO uplo  = BblasLower;
  enum BBLAS_DIAG diag  = BblasNonUnit;
  enum BBLAS_TRANS transA = BblasNoTrans;
  enum BBLAS_TRANS transB = BblasNoTrans;
  const double alpha = 2.0;
  const double beta = 0.0;
  const int batch_count = BATCH_COUNT;
  const int strideA = BATCH_COUNT;
  const int strideB = BATCH_COUNT;
  int seed[4] = {2, 4, 1, 7}; // random seed
  int colmaj = BblasColMajor; // Use column major ordering
  int info = 0;
  int lda;
  int ldb;
  double flops;
  
 #pragma omp parallel
  {
    if (omp_get_thread_num() == 0)
      {
	printf("OMP threads = %d\n", omp_get_num_threads());
      }
  }
  printf("MKL threads = %d\n", mkl_get_max_threads());
  printf("batch_count = %d\n", BATCH_COUNT);


  printf("M,N,perf(Cblas +OMP),perf(full intl), ratio(mkl/intl), perf(intl+conv), ratio(mkl/intl+conv) perf(blkintl),\
bsize, ratio(mkl/blkintl), perf(blkintl+conv), bsize+conv,\
ratio(mkl/(blkintl+conv)), error(intl), error(blkintl)\n");

  for (int M = 4; M < 32; M++){
    for (int N = 1; N <=M; N++){
      printf("M = %d\n", M);
      printf("N = %d\n", N);
      lda = M;
      ldb = M;
      flops = 1.0 * (N*M*M)*BATCH_COUNT;
      printf("GFlops = %f\n", flops/1e9);

      // Generate batch
      for (int idx = 0; idx < BATCH_COUNT; idx++)
	{
	  // Generate A
	  Ap2p[idx] = (double*) hbw_malloc(sizeof(double) * M*M);
	  LAPACKE_dlarnv_work(IONE, ISEED, M*M, Ap2p[idx]);
	  for (int i = 0; i < M; i++)
	    Ap2p[idx][i*M+i] +=1;

	  // Generate B
	  Bp2p[idx] = (double*) hbw_malloc(sizeof(double) * M*N);
	  Bref[idx] = (double*) hbw_malloc(sizeof(double) * M*N);
	  Bsol[idx] = (double*) hbw_malloc(sizeof(double) * M*N);
	  LAPACKE_dlarnv_work(IONE, ISEED, M*N, Bref[idx]);
	}

      //=================================================
      // Compute result using CBLAS
      //=================================================

      // Clear cache
      printf("Computing results using CBLAS (OpenMP)\n");
      time_mkl =0.0;
      for (int testid = 0; testid < nbtest; testid++){
	memcpy_bptp2ptp(Bp2p, Bref, M, N, batch_count);
	clearcache();    
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
	if(testid != 0) time_mkl += timediff;
      }
      time_mkl /= (nbtest-1);
      perf_mkl = flops / time_mkl / 1000;
      printf("CBLAS Time = %f us\n", time_mkl);
      printf("CBLAS Perf = %f GFlop/s\n\n",perf_mkl );
      
      //========================================
      //Convert to interleave layout
      //=======================================

      // Create interleaved matrices
      printf("Converting to interleaved format\n");
      arrayA = (double*)
        hbw_malloc(sizeof(double) * lda*M*batch_count);      
      arrayB = (double*)
        hbw_malloc(sizeof(double) * ldb*N*batch_count);
      
      // Convert Ap2p to interleaved layout
      time_cpyint = 0.0;
      for (int testid = 0; testid < nbconvtest; testid++) {
	clearcache();
	gettime();
	timediff = time;
	memcpy_aptp2intl(arrayA, Ap2p, M, batch_count);
	gettime();
	time_cpyint += (time - timediff);
      }
      time_cpyint /=nbconvtest;

      // Convert Bp2p to interleaved layout
      time_cpyB = 0.0;
      for (int testid = 0; testid < nbconvtest; testid++) {
	clearcache();
	gettime();
	timediff = time;
	memcpy_bptp2intl(arrayB, Bref, M, N, batch_count);
	gettime();
	time_cpyB += (time - timediff);
      }
      time_cpyB /=nbconvtest;
      time_cpyint += time_cpyB;

      //==============================================================
      // Calling full interleave kernel
      //==============================================================

      time_intl = 0;
      for (int testid = 0; testid < nbtest; testid++){
	memcpy_bptp2intl(arrayB, Bref, M, N, batch_count);
	clearcache();
	gettime();
	timediff = time;
	bblas_dtrsm_batch_intl(side, uplo, transA, diag,
			       M, N, alpha, (const double*)arrayA, strideA,
			       arrayB, strideB,
			       batch_count, info);
	gettime();
	timediff = time - timediff;
	if(testid != 0)time_intl += timediff;
      }
      time_intl /= (nbtest-1);

      //=======================================
      // convert solution back
      //=======================================

      // Convert interleaved B to p2p formt
      time_cpyB = 0.0;
      for (int testid = 0; testid < nbconvtest; testid++) {
	clearcache();
	gettime();
	timediff = time;
	memcpy_bintl2ptp(Bsol, arrayB, M, N, batch_count);
	gettime();
	time_cpyB += (time - timediff);
      }
      time_cpyB /=nbconvtest;
      time_cpyint += time_cpyB;
      
      perf_intl = flops / time_intl / 1000;
      perf_intl_conv = flops / (time_intl+ time_cpyint) / 1000;
      
      printf("INTL Time = %f us\n", time_intl);
      printf("INTL Perf = %f GFlop/s\n", perf_intl);
      printf("Ratio Time_mkl/Time_intl = %.2f\n", time_mkl/time_intl);
      printf("INTL Perf+conv = %f GFlop/s\n", perf_intl_conv);
      printf("Ratio Time_mkl/(Time_intl+conv) = %.2f\n", time_mkl/(time_intl + time_cpyint));

      // Calculate difference between results
      error_intl =  get_error(Bp2p, Bsol, M, N, batch_count);
      printf("INTL norm = %.2e\n\n", error_intl);
    
      //=========================================
      // Convert to block interleave layout
      //=========================================
      arrayAblk = (double*) 
	hbw_malloc(sizeof(double) * M*M*BATCH_COUNT); 
      arrayBblk = (double*)
	hbw_malloc(sizeof(double) * M*N*BATCH_COUNT);
      
      time_bestblkintl = 100000*time_mkl; //initialization
      best_block = 0;
      time_bestblkintl_conv = 100000*time_mkl; //initialization
      best_block_conv = 0;
      for (int BLOCK_SIZE = 16; BLOCK_SIZE <= MAX_BLOCK_SIZE; BLOCK_SIZE +=16) {

	// Create block interleaved
	printf("Converting to block interleaved format - block_size = %d \n", BLOCK_SIZE);
	int blocksrequired = batch_count / BLOCK_SIZE;
	int remainder = 0;
	if (batch_count % BLOCK_SIZE != 0)
	  {
	    blocksrequired += 1;
	    remainder = batch_count % BLOCK_SIZE;
	  }
	// Convert Ap2p to  block interleaved layout
	time_cpyblkint = 0.0;
	for (int testid = 0; testid < nbconvtest; testid++){
	  clearcache();
	  gettime();
	  timediff = time;
	  memcpy_aptp2blkintl(arrayAblk, Ap2p, M, BLOCK_SIZE, batch_count);
	  gettime();
	  time_cpyblkint += (time - timediff);
	}
	time_cpyblkint /=nbconvtest;

	// Convertt Bp2p to B block interleaved layout
	time_cpyB = 0.0;
	for (int testid = 0; testid < nbconvtest; testid++){
	  clearcache();
	  gettime();
	  timediff = time;
	  memcpy_bptp2blkintl(arrayBblk, Bref, M, N, BLOCK_SIZE, batch_count);
	  gettime();
	  time_cpyB += (time - timediff);
	}
	time_cpyB /=nbconvtest;
	time_cpyblkint += time_cpyB;
	
	//========================================================
	// Calling block interleave kernel
	//========================================================
	
      time_blkintl =0.0;
      for (int testid = 0; testid < nbtest; testid++){
	memcpy_bptp2blkintl(arrayBblk, Bref, M, N, BLOCK_SIZE, batch_count);
	clearcache();
	gettime();
	double timediff = time;
	bblas_dtrsm_batch_blkintl(
				  side, uplo, transA, diag,
				  M, N, alpha, (const double*) arrayAblk,
				  arrayBblk, BLOCK_SIZE,
				  batch_count, info);
	gettime();
	timediff = time - timediff;
	if(testid != 0)time_blkintl += timediff;
      }
      time_blkintl /=(nbtest-1);
      
      //=======================================================
      // Convert  B block interleaved layout back to p2p layout
      //======================================================
      time_cpyB = 0.0;
      for (int testid = 0; testid < nbconvtest; testid++){
	clearcache();
	gettime();
	timediff = time;
	memcpy_bblkintl2ptp(Bsol, arrayBblk, M, N, BLOCK_SIZE, batch_count);
	gettime();
	time_cpyB += (time - timediff);
      }
      time_cpyB /=nbconvtest;
      time_cpyblkint += time_cpyB;

      //Set best time and best block
      if ( time_blkintl < time_bestblkintl ) {
	time_bestblkintl = time_blkintl;
	best_block = BLOCK_SIZE;
      }
      if ( (time_blkintl + time_cpyblkint) < time_bestblkintl_conv ) {
	time_bestblkintl_conv = (time_blkintl + time_cpyblkint);
	best_block_conv = BLOCK_SIZE;
      }

      printf("M = %d\n", M);
      printf("N = %d\n", N);
      printf("BLKINTL Time = %f us\n", time_blkintl);
      printf("BLKINTLPerf = %f GFlop/s\n", flops / time_blkintl / 1000);
      printf("BLKINTLPerf+conv = %f GFlop/s\n", flops / (time_blkintl + time_cpyblkint) / 1000);
      printf("Ratio Time_mkl/Time_blkintl = %.2f\n", time_mkl/time_blkintl);
      printf("Ratio Time_mkl/(Time_blkintl + conv) = %.2f\n", time_mkl/(time_blkintl+ time_cpyblkint));

      error_blkintl =  get_error(Bp2p, Bsol, M, N, batch_count);
      printf("BLOCK INTL norm = %.2e\n\n", error_blkintl);
      }
      perf_blkintl = flops / time_bestblkintl / 1000;
      perf_blkintl_conv = flops / time_bestblkintl_conv / 1000;
      
      printf("%d,%d,%.2e,%.2e,%.2f,%.2e,%.2f,%.2e,%d,%.2f,%.2e,%d,%.2f,%.2e,%.2e\n", M,N, perf_mkl, perf_intl, perf_intl/perf_mkl,
	     perf_intl_conv, perf_intl_conv/perf_mkl, perf_blkintl, best_block, perf_blkintl/perf_mkl,
	     perf_blkintl_conv, best_block_conv, perf_blkintl_conv/perf_mkl, error_intl, error_blkintl);   

      // Hbw_Free memory
      hbw_free(arrayA);
      hbw_free(arrayB);
      hbw_free(arrayAblk);
      hbw_free(arrayBblk);
      for (int idx = 0; idx < batch_count; idx++)
	{
	  hbw_free(Ap2p[idx]);
	  hbw_free(Bp2p[idx]);
	  hbw_free(Bref[idx]);
	}
    }
  }
  
  hbw_free(Ap2p);
  hbw_free(Bp2p);
  hbw_free(Bref);
  hbw_free(bigA);
  hbw_free(bigB);
  hbw_free(bigC);
  return 0;
}

