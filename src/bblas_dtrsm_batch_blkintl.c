#include "bblas_interleaved.h"
#include<stdio.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_dtrsm_batch_blkintl(
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
		    
		      #pragma omp parallel for
		      for (int idx = 0; idx < remainder; idx++)
			{
			  if (k == 0 ) arrayB[startB + idx] *= alpha; // alpha B
			  
			  if (arrayB[startB + idx] != 0 ) {
			    arrayB[startB + idx] /= arrayA[(m*(m+1)/2)*blkidx*block_size +
							   ((2*m-k-1)*k/2 + k)*block_size + idx];
			  }
			  for (int i = k+1; i < m; i++)
			    {
			      if (k == 0 ) arrayB[m*n*blkidx*block_size + (j*m + i)*block_size + idx] *= alpha; // alpha B
			      arrayB[m*n*blkidx*block_size + (j*m + i)*block_size + idx] -=
				arrayB[startB + idx]*arrayA[(m*(m+1)/2)*blkidx*block_size +
							    ((2*m-k-1)*k/2 + i)*block_size + idx];
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
		    
		    #pragma omp parallel for
		    for (int idx = 0; idx < block_size; idx++)
		      {
			if (k == 0 ) arrayB[startB + idx] *= alpha; // alpha B
			
			if (arrayB[startB + idx] != 0 ) {
			  arrayB[startB + idx] /= arrayA[(m*(m+1)/2)*blkidx*block_size +
							 ((2*m-k-1)*k/2 + k)*block_size + idx];
			}
			for (int i = k+1; i < m; i++)
			  {
			    if (k == 0 ) arrayB[m*n*blkidx*block_size + (j*m + i)*block_size + idx] *= alpha; // alpha B
			    arrayB[m*n*blkidx*block_size + (j*m + i)*block_size + idx] -=
			      arrayB[startB + idx]*arrayA[(m*(m+1)/2)*blkidx*block_size +
							  ((2*m-k-1)*k/2 + i)*block_size + idx];
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
