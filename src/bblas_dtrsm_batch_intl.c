#include "bblas_interleaved.h"
#include<stdio.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_dtrsm_batch_intl(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    double alpha,
	const double *arrayA, int strideA,
    double *arrayB, int strideB,
    int batch_count, int info)
{
	// Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;
	// Note: arrayB(i,k,idx) = arrayB[k*strideA*M + i*strideA + idx]
  
  if ((side == BblasLeft)
      && (uplo == BblasLower)
      && (diag == BblasNonUnit)
      && (trans == BblasNoTrans) ) {
    
    for (int j = 0; j < n; j++)
      {
	for (int i = 0; i < m; i++) {
	  #pragma omp parallel for
	  for (int idx = 0; idx < batch_count; idx++) {
	    arrayB[(j*m+i)*strideB + idx] *= alpha;
	  }		  
	}
	for (int k = 0; k < m; k++)
	  {
	    int startB = (j*m + k)*strideB;

	    #pragma omp parallel for
	    for (int idx = 0; idx < batch_count; idx++)
	      {
		if (arrayB[startB + idx] != 0 ) {
		  arrayB[startB + idx] /= arrayA[((2*m-k-1)*k/2 + k)*strideA + idx];
		}
		for (int i = k+1; i < m; i++)
		  {
		    arrayB[(j*m + i)*strideB + idx] -=  arrayB[startB + idx]*arrayA[((2*m-k-1)*k/2 + i)*strideA + idx];
		  }
	      }
	  }
      }
  }
  info = BBLAS_SUCCESS;
}

#undef COMPLEX
