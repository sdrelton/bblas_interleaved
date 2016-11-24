#include "bblas_interleaved.h"
#include <mkl.h>
#include<stdio.h>

// Assumes interleaved in column major order

void bblas_dposv_intl(enum BBLAS_UPLO uplo,
                      int m, int n,
                      double **Ap2p, int lda,
                      double **Bp2p, int ldb,
                      double *work, int batch_count, int info)
{
	// Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;
	// Note: arrayB(i,k,idx) = arrayB[k*batch_count*M + i*batch_count + idx]
  
    double *arrayA = work;
    double *arrayB = (work + m*m*batch_count);
    double alpha = 1.0;
    // Convert Ap2p to interleaved layout
    memcpy_aptp2intl(arrayA, Ap2p, lda, batch_count);
  
    // Convert Bp2p to interleaved layout
    memcpy_bptp2intl(arrayB, Bp2p, ldb, n, batch_count);


    if (uplo != BblasLower) {
        printf("Configuration not implemented yet\n");
        return;
    }
    
    //Compute the Cholesky factorization A = L*L'
    bblas_dpotrf_intl_expert(CblasLower, m, arrayA,
                             batch_count, info);

    //================================================
    //Solve the system A*X = B, overwriting B with X.
    //================================================

    //Solve U'*X = B, overwriting B with X.
    bblas_dtrsm_intl_expert(BblasLeft, uplo, CblasNoTrans, CblasNonUnit,
                            m, n, alpha, arrayA, arrayB,
                            batch_count, info);

    //Solve U*X = B, overwriting B with X.
    bblas_dtrsm_intl_expert(BblasLeft, uplo, CblasTrans, CblasNonUnit,
                            m, n, alpha, arrayA, arrayB,
                            batch_count, info);
    // convert solution back
    memcpy_bintl2ptp(Bp2p, arrayB, m, n, batch_count);
    // convert factorization back
    memcpy_aintl2ptp(Ap2p, arrayA, n, batch_count);
    info = BBLAS_SUCCESS;
}

