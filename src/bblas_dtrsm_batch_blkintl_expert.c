#include "bblas_interleaved.h"
#include<stdio.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_dtrsm_batch_blkintl_expert(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    double alpha,
    const double * arrayA,
    double *arrayB, int block_size,
    int batch_count, int info)
{
  // Error checks go here
  // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
  //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;

  

  int numblocks = batch_count / block_size;
  int remainder = 0;
  if (batch_count % block_size != 0)
    {
      numblocks += 1;
      remainder = (batch_count % block_size);
    }
  
  if ((side == BblasLeft)
      && (uplo == BblasLower)
      && (diag == BblasNonUnit)
      && (trans == BblasNoTrans) ) {

       # pragma omp parallel for
      for (int blkidx = 0; blkidx < numblocks; blkidx++)
      {
          // Remainder
          if ((blkidx == numblocks-1) && (remainder != 0))
          {
              for (int j = 0; j < n; j++)
              {
                  for (int k = 0; k < m; k++)
                  {
                      int startB = m*n*blkidx*block_size + (j*m + k)*block_size;
                      int Akk = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 + k)*block_size;
                      int offset = k*block_size;

                      for (int i = k; i < m; i++)
                      {
                          int Bij = m*n*blkidx*block_size + (j*m +i)*block_size ;
                          int Aik = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 +i )*block_size;
			  #pragma ivdep
                          for (int idx = 0; idx < remainder; idx++)
                          {
                              if (k == 0 ) arrayB[Bij + idx] *= alpha; // alpha B
                              if (i == k ){
                                  arrayB[startB + idx] /= arrayA[Akk + idx];
                                  continue; 
                              }
                              arrayB[Bij + idx] -=arrayB[startB + idx]*arrayA[Aik + idx];
                          }
                      }
                  }
              }
          } else 
          {
              for (int j = 0; j < n; j++)
              {
                  for (int k = 0; k < m; k++)
                  {
                      int startB = m*n*blkidx*block_size + (j*m + k)*block_size;
                      int Akk = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 + k)*block_size;
                      int offset = k*block_size;
                      
                      for (int i = k; i < m; i++)
                      {
                          int Bij = m*n*blkidx*block_size + (j*m +i)*block_size ;
                          int Aik = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 +i )*block_size;
			  #pragma ivdep
                          for (int idx = 0; idx < block_size; idx++)
                          {
                              if (k == 0 ) arrayB[Bij + idx] *= alpha; // alpha B
                              if (i == k ){
                                  arrayB[startB + idx] /= arrayA[Akk + idx];
                                  continue; 
                              }
                              arrayB[Bij + idx] -=arrayB[startB + idx]*arrayA[Aik + idx];
                          }
                      }
                  }
              }
          }
      }
  }
  info = BBLAS_SUCCESS;
}

#undef COMPLEX
