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
#define MAX_BLOCK_SIZE 256
#define MAX_M 32
#define MIN_M 2
#define MAX_RHS 4
#define MIN_RHS 4
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
    double *work;
    double time_cpyblkint;  
    double time_blkintl;
    double time_bestblkintl;
    double time_bestblkintl_conv;
    double perf_blkintl;
    double perf_blkintl_conv;
    int best_block;
    int best_block_conv;

    // Now create pointer-to-pointer batch of random matrices
    double **Ap2p =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
    double **Aref =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
    double **Bp2p =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
    double **Xp2p =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
    double **Bref =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
    double **Bsol =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);

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

    printf("M,N,perf_lapack,perf(full intl),ratio(mkl/intl),perf(intl+conv),ratio(mkl/intl+conv),perf(blkintl),\
bsize,ratio(mkl/blkintl),perf_blkintl_conv,bsize+conv,\
ratio(mkl/(blkintl+conv)),error(intl)\n");
    
    for (int M = MIN_M; M <= MAX_M ; M++){
        for (int N = MIN_RHS; N <= MAX_RHS; N++){
            lda = M;
            ldb = M;
            flops = 2*(1.0 * (N*M*M)*BATCH_COUNT) + // 2TRSM
                (1.0/3.*M*M*M + 1./2.*M*M + 1./6.*M)*BATCH_COUNT; //POTRF
            
            // Generate batch
            for (int idx = 0; idx < BATCH_COUNT; idx++)
            {
                // Generate A
                Ap2p[idx] = (double*) hbw_malloc(sizeof(double) * M*M);
                Aref[idx] = (double*) hbw_malloc(sizeof(double) * M*M);
                LAPACKE_dlarnv_work(IONE, ISEED, M*M, Aref[idx]);
                for (int i = 0; i < M; i++)
                    Aref[idx][i*lda+i] += M;
                
                // Generate B
                Bp2p[idx] = (double*) hbw_malloc(sizeof(double) * M*N);
                Bref[idx] = (double*) hbw_malloc(sizeof(double) * M*N);
                Bsol[idx] = (double*) hbw_malloc(sizeof(double) * M*N);
                Xp2p[idx] = (double*) hbw_malloc(sizeof(double) * M*N);
                LAPACKE_dlarnv_work(IONE, ISEED, M*N, Bref[idx]);
            }
            
            //=================================================
            // Compute result using CBLAS
            //=================================================
            time_mkl =0.0;
            for (int testid = 0; testid < nbtest; testid++){
                memcpy_dbptp2ptp(Ap2p, Aref, M, M, batch_count);
                memcpy_dbptp2ptp(Bp2p, Bref, M, N, batch_count);
                clearcache();    
                gettime();
                timediff = time;
                
                #pragma omp parallel for
                for (int idx = 0; idx < batch_count; idx++)
                {
                    LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', M, N,
                                  Ap2p[idx], lda, Bp2p[idx], ldb);
                }
                
                gettime();
                timediff = time - timediff;
                if(testid != 0) time_mkl += timediff;
            }
            time_mkl /= (nbtest-1);
            perf_mkl = flops / time_mkl / 1000;
            
            //Copy the solution
            memcpy_dbptp2ptp(Xp2p, Bp2p, M, N, batch_count);
            
            //=============================================================
            // Compute with full interleave layout
            //=============================================================
            
            // Create interleaved matrices
            arrayA = (double*)
                hbw_malloc(sizeof(double) * lda*M*batch_count);
            arrayB = (double*)
                hbw_malloc(sizeof(double) * ldb*N*batch_count);
            
      
            // Calling full interleave kernel
            time_intl = 0;
            for (int testid = 0; testid < nbtest; testid++){
                /* // Convert Ap2p to interleaved layout */
                /* memcpy_daptp2intl(arrayA, Aref, M, batch_count); */
                /* // Convert Bp2p to interleaved layout */
                /* memcpy_dbptp2intl(arrayB, Bref, M, N, batch_count); */
                /* clearcache(); */
                /* gettime(); */
                /* timediff = time; */
                /* bblas_dposv_intl_expert(CblasLower, M, N, arrayA, */
                /*                         arrayB, batch_count, info); */
                gettime();
                timediff = time - timediff;
                if(testid != 0)time_intl += timediff;
            }
            time_intl /= (nbtest-1);
            
            // convert solution back
            //memcpy_dbintl2ptp(Bsol, arrayB, M, N, batch_count);
            //memcpy_daintl2ptp(Ap2p, arrayA, M, batch_count);
            //Performance computation
            perf_intl = flops / time_intl / 1000;
      
            // Copute forward error
            error_intl =  0.0;//get_derror(Xp2p, Bsol, M, N, batch_count);
            
            //=============================================
            // Compute with full interleave + conversion
            //=============================================
            work = (double*)
                hbw_malloc(sizeof(double) * (M+N)*M*batch_count);
            time_intl = 0;
            for (int testid = 0; testid < nbtest; testid++){
                // Copy Aref to Ap2p
                /* memcpy_dbptp2ptp(Ap2p, Aref, M, M, batch_count); */
                /* // Copy Bref to Bp2p */
                /* memcpy_dbptp2ptp(Bp2p, Bref, M, N, batch_count); */
                /* clearcache(); */
                /* gettime(); */
                /* timediff = time; */
                /* bblas_dposv_intl(uplo, M, N, Ap2p, lda, */
                /*                  Bp2p, ldb, */
                /*                  work, batch_count, info); */
                gettime();
                timediff = time - timediff;
                if(testid != 0)time_intl += timediff;
            }
            time_intl /= (nbtest-1);
            perf_intl_conv = flops / time_intl / 1000;
            
            // Copute forward error
            double error_intl_conv = 0.0;//get_derror(Xp2p, Bp2p, M, N, batch_count);
            
            //Free work
            hbw_free(work);

            //=========================================
            // Convert to block interleave layout
            //=========================================
            time_bestblkintl = 100000*time_mkl; //initialization
            best_block = 0;
            time_bestblkintl_conv = 100000*time_mkl; //initialization
            best_block_conv = 0;
            
            for (int BLOCK_SIZE = 2; BLOCK_SIZE <= MAX_BLOCK_SIZE; BLOCK_SIZE +=2) {
                
                // Create block interleaved
                int blocksrequired = batch_count / BLOCK_SIZE;
                int remainder = 0;
                if (batch_count % BLOCK_SIZE != 0)
                {
                    blocksrequired += 1;
                    remainder = batch_count % BLOCK_SIZE;
                }
	
                arrayAblk = (double*)
                    hbw_malloc(sizeof(double) * M*M*BLOCK_SIZE*blocksrequired);
                arrayBblk = (double*)
                    hbw_malloc(sizeof(double) * M*N*BLOCK_SIZE*blocksrequired);
                work = (double*) hbw_malloc(sizeof(double) * M*(N+M)*BLOCK_SIZE*blocksrequired);
      
		
                //========================================================
                // Calling block interleave kernel
                //========================================================
	
                time_blkintl =0.0;
                for (int testid = 0; testid < nbtest; testid++){
                    // Convert Ap2p to  block interleaved layout
                    /* memcpy_daptp2blkintl(arrayAblk, Aref, M, BLOCK_SIZE, batch_count); */
                    /* memcpy_dbptp2blkintl(arrayBblk, Bref, M, N, BLOCK_SIZE, batch_count); */
                    clearcache(); 
                    /* gettime(); */
                    /* double timediff = time; */
                    /* bblas_dposv_blkintl_expert(uplo, M, N, arrayAblk, */
                    /*                            arrayBblk, BLOCK_SIZE, */
                    /*                            batch_count, info); */
                    gettime();
                    timediff = time - timediff;
                    if(testid != 0)time_blkintl += timediff;
                }
                time_blkintl /=(nbtest-1);
                
                // Convert  B block interleaved layout back to p2p layout
                // memcpy_dbblkintl2ptp(Bsol, arrayBblk, M, N, BLOCK_SIZE, batch_count);
                
                //Set best time and best block
                if ( time_blkintl < time_bestblkintl ) {
                    time_bestblkintl = time_blkintl;
                    best_block = BLOCK_SIZE;
                }
                // Compute error
                error_blkintl = 0.0; // get_derror(Xp2p, Bsol, M, N, batch_count);
                

                //===========================================
                //Block interleave with internal conversion
                //==========================================
                double time_blkintl =0.0;
                for (int testid = 0; testid < nbtest; testid++){
                    memcpy_dbptp2ptp(Ap2p, Aref, M, M, batch_count);
                    memcpy_dbptp2ptp(Bp2p, Bref, M, N, batch_count);
                    clearcache();    
                    gettime();
                    timediff = time;
                    bblas_dposv_blkintl(uplo, M, N, Ap2p, lda, Bp2p, ldb,
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
            double error_blkintl_conv = get_derror(Xp2p, Bp2p, M, N, batch_count);
            
            printf("%d,%d,%.2e,%.2e,%.2f,%.2e,%.2f,%.2e,%d,%.2f,%.2e,%d,%.2f,%.2e\n", M,N, perf_mkl, perf_intl, perf_intl/perf_mkl,
                   perf_intl_conv, perf_intl_conv/perf_mkl, perf_blkintl, best_block, perf_blkintl/perf_mkl,
                   perf_blkintl_conv, best_block_conv, perf_blkintl_conv/perf_mkl, error_blkintl_conv);   
            
            
            //Hbw_Free memory
            hbw_free(arrayA);
            hbw_free(arrayB);
            for (int idx = 0; idx < BATCH_COUNT; idx++)
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


