#include "bblas_interleaved.h"
#include<stdio.h>
#include <mkl.h>

// Assumes interleaved in column major order

void bblas_dposv_intl_expert(enum BBLAS_UPLO uplo,
                             int m,
                             int n,
                             double *arrayA,
                             double *arrayB,
                             int batch_count, int info)
{
	// Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;
	// Note: arrayB(i,k,idx) = arrayB[k*batch_count*M + i*batch_count + idx]
  
    if (uplo != BblasLower) {
        printf("Configuration not implemented yet\n");
        return;
    }
    
    int lda = m;
    int alpha = 1.0;
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
      info = BBLAS_SUCCESS;
}
#undef COMPLEX
