#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>
#include <hbwmalloc.h>

#define nbtest 20
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
    double **Bp2p =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
    double **Xp2p =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
    double **Bref =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);
    double **Bsol =
        (double**) hbw_malloc(sizeof(double*)*BATCH_COUNT);

    // Setup parameters
    enum BBLAS_SIDE  side = BblasLeft;
    enum BBLAS_UPLO uplo  = BblasLower;
    enum BBLAS_DIAG diag  = BblasNonUnit;
    const double alpha = 2.0;
    const double beta = 0.0;
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

    printf("M,N,perf(Cblas +OMP),perf(full intl), ratio(mkl/intl), perf(intl+conv), ratio(mkl/intl+conv) perf(blkintl),\
bsize, ratio(mkl/blkintl), perf(blkintl+conv), bsize+conv,\
ratio(mkl/(blkintl+conv)), error(intl)\n");

    for (int M = 2; M < 33; M++){
        for (int N = 1; N <= MAX_RHS; N++){
            lda = M;
            ldb = M;
            flops = 1.0 * (N*M*M)*BATCH_COUNT;

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
                Xp2p[idx] = (double*) hbw_malloc(sizeof(double) * M*N);
                LAPACKE_dlarnv_work(IONE, ISEED, M*N, Bref[idx]);
            }

            //=================================================
            // Compute result using CBLAS
            //=================================================
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
                        BblasColMajor, side, uplo, CblasTrans, diag,
                        M, N, alpha, (const double**)Ap2p[idx], lda, Bp2p[idx], ldb);
                }
      
                gettime();
                timediff = time - timediff;
                if(testid != 0) time_mkl += timediff;
            }
            time_mkl /= (nbtest-1);
            perf_mkl = flops / time_mkl / 1000;
      
            //Copy the solution
            memcpy_bptp2ptp(Xp2p, Bp2p, M, N, batch_count);

            //=============================================================
            // Compute with full interleave layout
            //=============================================================

            // Create interleaved matrices
            arrayA = (double*)
                hbw_malloc(sizeof(double) * lda*M*batch_count);      
            arrayB = (double*)
                hbw_malloc(sizeof(double) * ldb*N*batch_count);
            work = (double*)
                hbw_malloc(sizeof(double) * (M+N)*M*batch_count);
      
            // Convert Ap2p to interleaved layout
            memcpy_aptp2intl(arrayA, Ap2p, M, batch_count);
      
            // Calling full interleave kernel
            time_intl = 0;
            for (int testid = 0; testid < nbtest; testid++){
                // Convert Bp2p to interleaved layout
                memcpy_bptp2intl(arrayB, Bref, M, N, batch_count);
                clearcache();
                gettime();
                timediff = time;
                bblas_dtrsm_intl_expert(side, uplo, CblasTrans, diag,
                                        M, N, alpha, (const double*)arrayA,
                                        arrayB, batch_count, info);
                gettime();
                timediff = time - timediff;
                if(testid != 0)time_intl += timediff;
            }
            time_intl /= (nbtest-1);

            // convert solution back
            memcpy_bintl2ptp(Bsol, arrayB, M, N, batch_count);

            //Performance computation
            perf_intl = flops / time_intl / 1000;
      
            // Copute forward error
            error_intl =  get_error(Xp2p, Bsol, M, N, batch_count);
      
            //=============================================
            // Compute with full interleave + conversion
            //=============================================
      
            time_intl = 0;
            for (int testid = 0; testid < nbtest; testid++){
                // Copy Bref to Bp2p 
                memcpy_bptp2ptp(Bp2p, Bref, M, N, batch_count);
                clearcache();
                gettime();
                timediff = time;
                bblas_dtrsm_intl(side, uplo, CblasTrans, diag,
                                 M, N, alpha, Ap2p, lda,
                                 Bp2p, ldb,
                                 work, batch_count, info);
                gettime();
                timediff = time - timediff;
                if(testid != 0)time_intl += timediff;
            }
            time_intl /= (nbtest-1);
            perf_intl_conv = flops / time_intl / 1000;
      
            // Copute forward error
            double error_intl_conv =  get_error(Xp2p, Bp2p, M, N, batch_count);

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
	
                arrayAblk = (double*) 
                    hbw_malloc(sizeof(double) * M*M*BLOCK_SIZE*blocksrequired); 
                arrayBblk = (double*)
                    hbw_malloc(sizeof(double) * M*N*BLOCK_SIZE*blocksrequired);
                work = (double*) hbw_malloc(sizeof(double) * M*(N+M)*BLOCK_SIZE*blocksrequired);
      
                // Convert Ap2p to  block interleaved layout
                memcpy_aptp2blkintl(arrayAblk, Ap2p, M, BLOCK_SIZE, batch_count);
	
	
                //========================================================
                // Calling block interleave kernel
                //========================================================
	
                time_blkintl =0.0;
                for (int testid = 0; testid < nbtest; testid++){
                    memcpy_bptp2blkintl(arrayBblk, Bref, M, N, BLOCK_SIZE, batch_count);
                    clearcache();
                    gettime();
                    double timediff = time;
                    bblas_dtrsm_blkintl_expert(
                        side, uplo, CblasTrans, diag,
                        M, N, alpha, (const double*) arrayAblk,
                        arrayBblk, BLOCK_SIZE,
                        batch_count, info);
                    gettime();
                    timediff = time - timediff;
                    if(testid != 0)time_blkintl += timediff;
                }
                time_blkintl /=(nbtest-1);
	
                // Convert  B block interleaved layout back to p2p layout
                memcpy_bblkintl2ptp(Bsol, arrayBblk, M, N, BLOCK_SIZE, batch_count);
	
                //Set best time and best block
                if ( time_blkintl < time_bestblkintl ) {
                    time_bestblkintl = time_blkintl;
                    best_block = BLOCK_SIZE;
                }
                // Compute error
                error_blkintl =  get_error(Xp2p, Bsol, M, N, batch_count);
	
                //===========================================
                //Block interleave with internal conversion
                //==========================================
                double time_blkintl =0.0;
                for (int testid = 0; testid < nbtest; testid++){
                    memcpy_bptp2ptp(Bp2p, Bref, M, N, batch_count);
                    clearcache();    
                    gettime();
                    timediff = time;
                    bblas_dtrsm_blkintl(
                        side, uplo, CblasTrans, diag,
                        M, N, alpha, Ap2p, lda, Bp2p, ldb,
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
            double error_blkintl_conv =  get_error(Xp2p, Bp2p, M, N, batch_count);

            printf("%d,%d,%.2e,%.2e,%.2f,%.2e,%.2f,%.2e,%d,%.2f,%.2e,%d,%.2f,%.2e\n", M,N, perf_mkl, perf_intl, perf_intl/perf_mkl,
                   perf_intl_conv, perf_intl_conv/perf_mkl, perf_blkintl, best_block, perf_blkintl/perf_mkl,
                   perf_blkintl_conv, best_block_conv, perf_blkintl_conv/perf_mkl, error_blkintl_conv);   
      

            // Hbw_Free memory
            hbw_free(arrayA);
            hbw_free(arrayB);
            for (int idx = 0; idx < batch_count; idx++)
            {
                hbw_free(Ap2p[idx]);
                hbw_free(Bp2p[idx]);
                hbw_free(Bref[idx]);
                hbw_free(Bsol[idx]);
                hbw_free(Xp2p[idx]);
            }
        }
    }
  
    hbw_free(Ap2p);
    hbw_free(Bp2p);
    hbw_free(Bref);
    hbw_free(Bsol);
    hbw_free(Xp2p);
    free(bigA);
    free(bigB);
    return 0;
}


