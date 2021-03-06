#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>
#include <hbwmalloc.h>

#define nbtest 10
#define BATCH_COUNT 10000
#define MAX_BLOCK_SIZE 720
#define MAX_M 64
#define MIN_M 2
#define MAX_RHS 1
#define CACHECLEARSIZE 20000000
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
    float *arrayA = NULL;
    float *arrayB = NULL;
    double time_cpyint;
    double time_cpyB;
    float norm;
    float error_intl;
    float error_blkintl;
    double perf_intl;
    double perf_intl_conv;
    int startpos;
    int ctr;

    //Block interleave variables
    float *arrayAblk;
    float *arrayBblk;
    float *work;
    double time_cpyblkint;  
    double time_blkintl;
    double time_bestblkintl;
    double time_bestblkintl_conv;
    double perf_blkintl;
    double perf_blkintl_conv;
    int best_block;
    int best_block_conv;

    // Now create pointer-to-pointer batch of random matrices
    float **Ap2p =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
    float **Aref =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
    float **Bp2p =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
    float **Xp2p =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
    float **Bref =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
    float **Bsol =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);

    // Setup parameters
    enum BBLAS_UPLO uplo  = CblasLower;
    const int batch_count = BATCH_COUNT;
    int info = 0;
    int lda;
    int ldb;
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

    printf("M,N,perf(LAPACKE +OMP),perf(full intl), ratio(mkl/intl), perf(intl+conv), ratio(mkl/intl+conv) perf(blkintl),\
bsize, ratio(mkl/blkintl), perf(blkintl+conv), bsize+conv,\
ratio(mkl/(blkintl+conv)), error(intl)\n");

    for (int M = MIN_M; M <= MAX_M ; M++){
        for (int N = 1; N <= MAX_RHS; N++){
            lda = M;
            ldb = M;
            flops = 1.0 * (N*M*M)*BATCH_COUNT;

            // Generate batch
            for (int idx = 0; idx < BATCH_COUNT; idx++)
            {
                // Generate A
                Ap2p[idx] = (float*) hbw_malloc(sizeof(float) * M*M);
                Aref[idx] = (float*) hbw_malloc(sizeof(float) * M*M);
                LAPACKE_slarnv_work(IONE, ISEED, M*M, Aref[idx]);
                for (int i = 0; i < M; i++)
                    Aref[idx][i*lda+i] += M;

                // Generate B
                Bp2p[idx] = (float*) hbw_malloc(sizeof(float) * M*N);
                Bref[idx] = (float*) hbw_malloc(sizeof(float) * M*N);
                Bsol[idx] = (float*) hbw_malloc(sizeof(float) * M*N);
                Xp2p[idx] = (float*) hbw_malloc(sizeof(float) * M*N);
                LAPACKE_slarnv_work(IONE, ISEED, M*N, Bref[idx]);
            }

            //=================================================
            // Compute result using CBLAS
            //=================================================
            time_mkl =0.0;
            for (int testid = 0; testid < nbtest; testid++){
                memcpy_sbptp2ptp(Ap2p, Aref, M, M, batch_count);
                memcpy_sbptp2ptp(Bp2p, Bref, M, N, batch_count);
                clearcache();    
                gettime();
                timediff = time;
      
                #pragma omp parallel for
                for (int idx = 0; idx < batch_count; idx++)
                {
                    LAPACKE_sposv(LAPACK_COL_MAJOR, 'L', M, N,
                                  Ap2p[idx], lda, Bp2p[idx], ldb);
                }
      
                gettime();
                timediff = time - timediff;
                if(testid != 0) time_mkl += timediff;
            }
            time_mkl /= (nbtest-1);
            perf_mkl = flops / time_mkl / 1000;
      
            //Copy the solution
            memcpy_sbptp2ptp(Xp2p, Bp2p, M, N, batch_count);
		
            //=============================================================
            // Compute with full interleave layout
            //=============================================================

            // Create interleaved matrices
            arrayA = (float*)
                hbw_malloc(sizeof(float) * lda*M*batch_count);      
            arrayB = (float*)
                hbw_malloc(sizeof(float) * ldb*N*batch_count);
      
      
            // Calling full interleave kernel
            time_intl = 0;
            for (int testid = 0; testid < nbtest; testid++){
	      // Convert Ap2p to interleaved layout
	      memcpy_saptp2intl(arrayA, Aref, M, batch_count);
	      // Convert Bp2p to interleaved layout
	      memcpy_sbptp2intl(arrayB, Bref, M, N, batch_count);
	      clearcache();
	      gettime();
	      timediff = time;
	      bblas_sposv_intl_expert(CblasLower, M, N, arrayA,
				      arrayB, batch_count, info);
                gettime();
                timediff = time - timediff;
                if(testid != 0)time_intl += timediff;
            }
            time_intl /= (nbtest-1);

            // convert solution back
            memcpy_sbintl2ptp(Bsol, arrayB, M, N, batch_count);
            memcpy_saintl2ptp(Ap2p, arrayA, M, batch_count);
            //Performance computation
            perf_intl = flops / time_intl / 1000;
      
            // Copute forward error
            error_intl =  get_serror(Xp2p, Bsol, M, N, batch_count);

            //=============================================
            // Compute with full interleave + conversion
            //=============================================
	    work = (float*)
	      hbw_malloc(sizeof(float) * (M+N)*M*batch_count);
            time_intl = 0;
            for (int testid = 0; testid < nbtest; testid++){
	      // Copy Aref to Ap2p 
	      memcpy_sbptp2ptp(Ap2p, Aref, M, M, batch_count);
                // Copy Bref to Bp2p 
                memcpy_sbptp2ptp(Bp2p, Bref, M, N, batch_count);
                clearcache();
                gettime();
                timediff = time;
                bblas_sposv_intl(uplo, M, N, Ap2p, lda,
                                 Bp2p, ldb,
                                 work, batch_count, info);
                gettime();
                timediff = time - timediff;
                if(testid != 0)time_intl += timediff;
            }
            time_intl /= (nbtest-1);
            perf_intl_conv = flops / time_intl / 1000;
      
            // Copute forward error
            float error_intl_conv =  get_serror(Xp2p, Bp2p, M, N, batch_count);

            //Free work
            hbw_free(work);

            //=========================================
            // Convert to block interleave layout
            //=========================================
            time_bestblkintl = 100000*time_mkl; //initialization
            best_block = 0;
            time_bestblkintl_conv = 100000*time_mkl; //initialization
            best_block_conv = 0;

            for (int BLOCK_SIZE = 8; BLOCK_SIZE <= MAX_BLOCK_SIZE; BLOCK_SIZE +=8) {
	
                // Create block interleaved
                int blocksrequired = batch_count / BLOCK_SIZE;
                int remainder = 0;
                if (batch_count % BLOCK_SIZE != 0)
                {
                    blocksrequired += 1;
                    remainder = batch_count % BLOCK_SIZE;
                }
	
                arrayAblk = (float*) 
                    hbw_malloc(sizeof(float) * M*M*BLOCK_SIZE*blocksrequired); 
                arrayBblk = (float*)
                    hbw_malloc(sizeof(float) * M*N*BLOCK_SIZE*blocksrequired);
                work = (float*) hbw_malloc(sizeof(float) * M*(N+M)*BLOCK_SIZE*blocksrequired);
      
		
                //========================================================
                // Calling block interleave kernel
                //========================================================
	
                time_blkintl =0.0;
                for (int testid = 0; testid < nbtest; testid++){
                // Convert Ap2p to  block interleaved layout
		  memcpy_saptp2blkintl(arrayAblk, Aref, M, BLOCK_SIZE, batch_count);
		  memcpy_sbptp2blkintl(arrayBblk, Bref, M, N, BLOCK_SIZE, batch_count);
		  clearcache();
		  gettime();
                    double timediff = time;
                    bblas_sposv_blkintl_expert(uplo, M, N, arrayAblk,
					       arrayBblk, BLOCK_SIZE,
					       batch_count, info);
                    gettime();
                    timediff = time - timediff;
                    if(testid != 0)time_blkintl += timediff;
                }
                time_blkintl /=(nbtest-1);
	
                // Convert  B block interleaved layout back to p2p layout
                memcpy_sbblkintl2ptp(Bsol, arrayBblk, M, N, BLOCK_SIZE, batch_count);
	
                //Set best time and best block
                if ( time_blkintl < time_bestblkintl ) {
                    time_bestblkintl = time_blkintl;
                    best_block = BLOCK_SIZE;
                }
                // Compute error
                error_blkintl =  get_serror(Xp2p, Bsol, M, N, batch_count);
	
                //===========================================
                //Block interleave with internal conversion
                //==========================================
                double time_blkintl =0.0;
                for (int testid = 0; testid < nbtest; testid++){
                    memcpy_sbptp2ptp(Ap2p, Aref, M, M, batch_count);
                    memcpy_sbptp2ptp(Bp2p, Bref, M, N, batch_count);
                    clearcache();    
                    gettime();
                    timediff = time;
                    bblas_sposv_blkintl(uplo, M, N, Ap2p, lda, Bp2p, ldb,
					BLOCK_SIZE, work, batch_count, info);
                    gettime();
                    timediff = time - timediff;
                    if(testid != 0) time_blkintl += timediff;
                }
                time_blkintl /= (nbtest-1);
		
                //Set best time and best block (with internal conversion)
                if ( time_blkintl < time_bestblkintl_conv ) {
		  time_bestblkintl_conv = time_blkintl;
		  best_block_conv = BLOCK_SIZE;
                }
                hbw_free(arrayAblk);
                hbw_free(arrayBblk);
                hbw_free(work);
            }
	    
            perf_blkintl = flops / time_bestblkintl / 1000;
            perf_blkintl_conv = flops / time_bestblkintl_conv / 1000;
	    
            // Calculate difference between results
            float error_blkintl_conv =  get_serror(Xp2p, Bp2p, M, N, batch_count);
	    
            printf("%d,%d,%.2e,%.2e,%.2f,%.2e,%.2f,%.2e,%d,%.2f,%.2e,%d,%.2f,%.2e\n", M,N, perf_mkl, perf_intl, perf_intl/perf_mkl,
                   perf_intl_conv, perf_intl_conv/perf_mkl, perf_blkintl, best_block, perf_blkintl/perf_mkl,
                   perf_blkintl_conv, best_block_conv, perf_blkintl_conv/perf_mkl, error_blkintl_conv);   
      

            // Hbw_Free memory
            hbw_free(arrayA);
            hbw_free(arrayB);
            for (int idx = 0; idx < batch_count; idx++)
            {
                hbw_free(Ap2p[idx]);
                hbw_free(Aref[idx]);
                hbw_free(Bp2p[idx]);
                hbw_free(Bref[idx]);
                hbw_free(Bsol[idx]);
                hbw_free(Xp2p[idx]);
            }
        }
    }
  
    hbw_free(Ap2p);
    hbw_free(Aref);
    hbw_free(Bp2p);
    hbw_free(Bref);
    hbw_free(Bsol);
    hbw_free(Xp2p);
    free(bigA);
    free(bigB);
    return 0;
}


