#include "bblas_interleaved.h"
#include<stdio.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_dtrsm_intl(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    double alpha,
    const double **Ap2p, int lda,
    double **Bp2p, int ldb,
    double *work, int batch_count, int info)
{
  // Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;
  // Note: arrayB(i,k,idx) = arrayB[k*strideA*M + i*strideA + idx]
  

  double *arrayA = work;
  double *arrayB = (work + m*m*batch_count);
  int strideB = batch_count;
  int strideA = batch_count;

  // Convert Ap2p to interleaved layout
  memcpy_aptp2intl(arrayA, Ap2p, m, batch_count);
  
  // Convert Bp2p to interleaved layout
  memcpy_bptp2intl(arrayB, Bp2p, m, n, batch_count);
  
  if ((side == BblasLeft)
      && (uplo == BblasLower)
      && (diag == BblasNonUnit)
      && (trans == BblasNoTrans) ) {
    
    for (int j = 0; j < n; j++)
      {
	for (int k = 0; k < m; k++)
	  {
	    int startB = (j*m + k)*strideB;
	    int Akk = ((2*m-k-1)*k/2 + k)*strideA;
	    for (int i = k; i < m; i++) {
	      int Bij = (j*m+i)*strideB;
	      int Aik = ((2*m-k-1)*k/2 + i)*strideA;
              #pragma omp parallel for simd
	      #pragma ivdep
	      for (int idx = 0; idx < batch_count; idx++)
		{
		  if (k == 0 ) arrayB[ Bij + idx] *= alpha; // alpha B
		  if (i == k) {
		    arrayB[startB + idx] /= arrayA[Akk + idx];
		    continue;
		      } 
		  arrayB[Bij + idx] -=  arrayB[startB + idx]*arrayA[ Aik + idx];
		}
	    }
	  }
      }
  }
  // convert solution back
  memcpy_bintl2ptp(Bp2p, arrayB, m, n, batch_count);
  info = BBLAS_SUCCESS;
}

#undef COMPLEX
