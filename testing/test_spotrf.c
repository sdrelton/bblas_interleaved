#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include <mkl.h>
#include <math.h>
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

    // Timer and perf variables
    double time, timediff, time_mkl;
    double time_intl, time_intl_conv;
    double time_blkintl, time_blkintl_conv;
    double perf_mkl, perf_intl, perf_intl_conv; 
    double perf_blkintl, perf_blkintl_conv;
    struct timeval tv;
    
    float *arrayA = NULL;
    float *arrayAblk = NULL;
    float *work;

    // Now create pointer-to-pointer batch of random matrices
    float **Ap2p =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
    float **Aref =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);
    float **Asol =
        (float**) hbw_malloc(sizeof(float*)*BATCH_COUNT);

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

    printf("N,perf(LAPACK +OMP),perf(full intl), ratio(mkl/intl), perf(intl+conv), ratio(mkl/intl+conv) perf(blkintl),\
bsize, ratio(mkl/blkintl), perf(blkintl+conv), bsize+conv,		\
ratio(mkl/(blkintl+conv)), error(intl)\n");
  
    for (int N = 2; N < 33; N++){
        lda = N;
        flops = (1.0/3.*N*N*N + 1./2.*N*N + 1./6.*N)*BATCH_COUNT;
      
        // Generate batch
        for (int idx = 0; idx < BATCH_COUNT; idx++)
        {
            // Generate A
            Ap2p[idx] = (float*) hbw_malloc(sizeof(float) * N*N);
            Asol[idx] = (float*) hbw_malloc(sizeof(float) * N*N);
            Aref[idx] = (float*) hbw_malloc(sizeof(float) * N*N);
            LAPACKE_slarnv_work(IONE, ISEED, N*N, Aref[idx]);
            for (int i = 0; i < N; i++)
                Aref[idx][i*lda+i] +=N;
        }

        //=================================================
        // Compute result using LAPACKE
        //=================================================
        time_mkl =0.0;
        for (int testid = 0; testid < nbtest; testid++){
            memcpy_sbptp2ptp(Ap2p, Aref, N, N, batch_count);
            clearcache();    
            gettime();
            timediff = time;
            #pragma omp parallel for
            for (int idx = 0; idx < batch_count; idx++) {
                LAPACKE_spotrf(LAPACK_COL_MAJOR, 'L', N, Ap2p[idx], lda);
            }
            gettime();
            timediff = time - timediff;
            if(testid != 0) time_mkl += timediff;
        }
        time_mkl /= (nbtest-1);
        perf_mkl = flops / time_mkl / 1000;
      
        //Copy the solution
        memcpy_sbptp2ptp(Asol, Ap2p, N, N, batch_count);

        //=============================================================
        // Compute with full interleave layout
        //=============================================================
	
        // Create interleaved matrices
        arrayA = (float*)
            hbw_malloc(sizeof(float) * lda*N*batch_count);      
	
        // Calling full interleave kernel
        time_intl = 0;
        for (int testid = 0; testid < nbtest; testid++){
            // Convert Ap2p to interleaved layout
            memcpy_saptp2intl(arrayA, Aref, N, batch_count);
            clearcache();
            gettime();
            timediff = time;
            bblas_spotrf_intl_expert(CblasLower, N, arrayA,
                                     batch_count, info);
            gettime();
            timediff = time - timediff;
            if(testid != 0)time_intl += timediff;
        }
        time_intl /= (nbtest-1);
        // convert solution back
        memcpy_saintl2ptp(Ap2p, arrayA, N, batch_count);
	
        //Performance computation
        perf_intl = flops / time_intl / 1000;
	
        // Copute forward error
        float error_intl =  get_serror(Asol, Ap2p, N, N, batch_count);

        //=======================================================
        // Compute with full interleave with internal conversion
        //======================================================
        work = (float*)
            hbw_malloc(sizeof(float) * lda*N*batch_count);
	
        time_intl = 0;
        for (int testid = 0; testid < nbtest; testid++){
            // Copy Aref to Ap2p 
            memcpy_sbptp2ptp(Ap2p, Aref, N, N, batch_count);
            clearcache();
            gettime();
            timediff = time;
            bblas_spotrf_intl(CblasLower, N, Ap2p, lda,
                              work,  batch_count, info); 
            gettime();
            timediff = time - timediff;
            if(testid != 0)time_intl += timediff;
        }
        time_intl /= (nbtest-1);
        perf_intl_conv = flops / time_intl / 1000;
	
        // Copute forward error
        float error_intl_conv =  get_serror(Asol, Ap2p, N, N, batch_count);
        //Free work
        hbw_free(work);
    
        float error_blkintl;
        float error_blkintl_conv;
        double time_bestblkintl = 100000*time_mkl; //initialization
        double time_bestblkintl_conv = 100000*time_mkl; //initialization
        int best_block = 0;
        int best_block_conv = 0;
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
                hbw_malloc(sizeof(float) * N*N*BLOCK_SIZE*blocksrequired); 
            work = (float*) hbw_malloc(sizeof(float) *N*N*BLOCK_SIZE*blocksrequired);

            //========================================================
            // Calling block interleave kernel
            //========================================================
            time_blkintl =0.0;
            for (int testid = 0; testid < nbtest; testid++){
                // Convert Ap2p to  block interleaved layout
                memcpy_saptp2blkintl(arrayAblk, Aref, N, BLOCK_SIZE, batch_count);
                clearcache();
                gettime();
                double timediff = time;
                bblas_spotrf_blkintl_expert(CblasLower, N, arrayAblk, lda,
                                            BLOCK_SIZE, batch_count, info); 
                gettime();
                timediff = time - timediff;
                if(testid != 0)time_blkintl += timediff;
            }
            time_blkintl /=(nbtest-1);
	  
            // Convert  A block interleaved layout back to p2p layout
            memcpy_sablkintl2ptp(Ap2p, arrayAblk, N, BLOCK_SIZE, batch_count);
	  
            //Set best time and best block
            if ( time_blkintl < time_bestblkintl ) {
                time_bestblkintl = time_blkintl;
                best_block = BLOCK_SIZE;
            }
            perf_blkintl = flops / time_bestblkintl / 1000;
            // Compute error
            error_blkintl =  get_serror(Ap2p, Asol, N, N, batch_count);

            //===========================================
            //Block interleave with internal conversion
            //==========================================
            time_blkintl =0.0;
            for (int testid = 0; testid < nbtest; testid++){
                memcpy_sbptp2ptp(Ap2p, Aref, N, N, batch_count);
                clearcache();    
                gettime();
                timediff = time;
                bblas_spotrf_blkintl(CblasLower, N, Ap2p, lda,
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
                
            hbw_free(work);
        }
        perf_blkintl_conv = flops / time_bestblkintl_conv / 1000;
        error_blkintl_conv =  get_serror(Ap2p, Asol, N, N, batch_count);
            
        printf("%d,%.2e,%.2e,%.2f,%.2e,%.2f,%.2e,%d,%.2f,%.2e,%d,%.2f,%.2e\n", N, perf_mkl, perf_intl, perf_intl/perf_mkl,
               perf_intl_conv, perf_intl_conv/perf_mkl, perf_blkintl, best_block, perf_blkintl/perf_mkl,
               perf_blkintl_conv, best_block_conv, perf_blkintl_conv/perf_mkl, error_blkintl);   	
            
        // Hbw_Free memory
        hbw_free(arrayA);
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
    
