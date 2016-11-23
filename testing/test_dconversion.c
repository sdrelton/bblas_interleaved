#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <omp.h>
#include <math.h>


void memcpy_bptp2intl(double *arrayB, double **Bp2p, int m, int n, int batch_count){

  for (int pos = 0; pos < m*n; pos++)
    {
      int offset = pos*batch_count;
      #pragma omp parallel for
      #pragma ivdep
      for (int idx = 0; idx < batch_count; idx++)
	{
	  arrayB[offset + idx] = Bp2p[idx][pos];
	}
    }
  
}
void memcpy_bptp2blkintl(double *arrayBblk, double **Bp2p, int m, int n, int block_size, int batch_count) {

  int blocksrequired = batch_count / block_size;
  int remainder = 0;
  if (batch_count % block_size != 0)
    {
      blocksrequired += 1;
      remainder = batch_count % block_size;
    }

  #pragma omp parallel for
  #pragma ivdep
  for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
    {
      int  startpos = blkidx * block_size * m*n;
      if (blkidx == blocksrequired - 1 && remainder != 0)
	{
	  // Remainders
	  for (int pos = 0; pos < m*n; pos++) {
	    int offset = startpos + pos*block_size;
	    for (int idx = 0; idx < remainder; idx++)
	      {
		arrayBblk[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
	      }
	  }
	}
      else
	{
	  for (int pos = 0; pos < m*n; pos++)
	    {
	      int offset = startpos + pos*block_size;
	      for (int idx = 0; idx < block_size; idx++)
		{
		  arrayBblk[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
		}
	    }
	}
    }
}

void memcpy_aptp2intl(double *arrayA, double **Ap2p, int m, int batch_count) {
  int lda = m;
  for (int j = 0; j < m; j++) {
    for (int i = j; i < m; i++ ) {
      int offset = (j*(2*m-j-1)/2 + i)*batch_count;
      #pragma omp parallel for
      #pragma ivdep
      for (int idx = 0; idx < batch_count; idx++)
	{
	  arrayA[offset + idx] = Ap2p[idx][j*lda+i];
	}
    }
  }
  
}


void memcpy_aintl2ptp(double **Ap2p, double *arrayA, int m, int batch_count) {
  int lda = m;
  for (int j = 0; j < m; j++) {
    for (int i = j; i < m; i++ ) {
      int offset = (j*(2*m-j-1)/2 + i)*batch_count;
      #pragma omp parallel for
      #pragma ivdep
      for (int idx = 0; idx < batch_count; idx++)
	{
	  Ap2p[idx][j*lda+i] = arrayA[offset + idx];
	}
    }
  }
  
}


void memcpy_aptp2blkintl(double *arrayAblk, double **Ap2p, int m, int block_size, int batch_count) {

  int blocksrequired = batch_count / block_size;
  int remainder = 0;
  int lda = m;
  int startpos;
  if (batch_count % block_size != 0)
    {
      blocksrequired += 1;
      remainder = batch_count % block_size;
    }

  #pragma omp parallel for
  #pragma ivdep
  for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
    {
      startpos = blkidx * block_size * m*(m+1)/2;
      if ((blkidx == blocksrequired - 1) && (remainder != 0))
	{
	  // Remainders
	  for (int j = 0; j < m; j++)
	    for (int i = j; i < m; i++) {
	      int offset = startpos + (j*(2*m-j-1)/2 + i)*block_size;
	      for (int idx = 0; idx < remainder; idx++) {
		arrayAblk[offset + idx] = Ap2p[blkidx * block_size + idx][j*lda+i];
	      }
	    }
	}
      else
	{
	  for (int j = 0; j < m; j++) 
	    for (int i = j; i < m; i++ ) {
	      int offset = startpos + (j*(2*m-j-1)/2 + i)*block_size;
	      for (int idx = 0; idx < block_size; idx++)
		{
		  arrayAblk[offset + idx] = Ap2p[blkidx * block_size + idx][j*lda+i];
		}
	    }
	}
    }
}



void memcpy_ablkintl2ptp(double **Ap2p, double *arrayAblk, int m, int block_size, int batch_count) {

  int blocksrequired = batch_count / block_size;
  int remainder = 0;
  int lda = m;
  int startpos;
  if (batch_count % block_size != 0)
    {
      blocksrequired += 1;
      remainder = batch_count % block_size;
    }

  #pragma omp parallel for
  #pragma ivdep
  for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
    {
      startpos = blkidx * block_size * m*(m+1)/2;
      if ((blkidx == blocksrequired - 1) && (remainder != 0))
	{
	  // Remainders
	  for (int j = 0; j < m; j++)
	    for (int i = j; i < m; i++) {
	      int offset = startpos + (j*(2*m-j-1)/2 + i)*block_size;
	      for (int idx = 0; idx < remainder; idx++) {
		Ap2p[blkidx * block_size + idx][j*lda+i] = arrayAblk[offset + idx];
	      }
	    }
	}
      else
	{
	  for (int j = 0; j < m; j++) 
	    for (int i = j; i < m; i++ ) {
	      int offset = startpos + (j*(2*m-j-1)/2 + i)*block_size;
	      for (int idx = 0; idx < block_size; idx++)
		{
		  Ap2p[blkidx * block_size + idx][j*lda+i] = arrayAblk[offset + idx];
		}
	    }
	}
    }
}



void memcpy_bintl2ptp(double **Bp2p, double *arrayB, int m, int n, int batch_count) {

  for (int pos = 0; pos < m*n; pos++)
    {
      int offset = pos*batch_count;
      #pragma omp parallel for
      #pragma ivdep
      for (int idx = 0; idx < batch_count; idx++)
	{
	  Bp2p[idx][pos] = arrayB[offset + idx];
	}
    }
}


void memcpy_bblkintl2ptp(double **Bp2p, double *arrayBblk, int m, int n, int block_size, int batch_count) {

  int blocksrequired = batch_count / block_size;
  int remainder = 0;
  int startpos;
  #pragma omp parallel for
  #pragma ivdep
  for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
    {
      startpos = blkidx * block_size * m*n;
      if (blkidx == blocksrequired - 1 && remainder != 0)
	{
	  // Remainders
	  for (int pos = 0; pos < m*n; pos++) {
	    int offset = startpos + pos*block_size;
	    for (int idx = 0; idx < remainder; idx++)
	      {
		Bp2p[blkidx * block_size + idx][pos] = arrayBblk[offset + idx];
	      }
	  }
	}
      else
	{
	  for (int pos = 0; pos < m*n; pos++)
	    {
	      int offset = startpos + pos*block_size;
	      for (int idx = 0; idx < block_size; idx++)
		{
		  Bp2p[blkidx * block_size + idx][pos] = arrayBblk[offset + idx];
		}
	    }
	}
    }  
}

void memcpy_bptp2ptp(double **Bp2p, double **Bref, int m, int n, int batch_count) {
  
  for (int idx = 0; idx < batch_count; idx++)
    {
      memcpy(Bp2p[idx], Bref[idx], sizeof(double)*m*n);      
    }
}


double get_error(double **Bref, double **Bsol, int m, int n, int batch_count) {

  double error = 0.0;
  double tmp_error;

  for (int idx = 0; idx < batch_count; idx++) {
    tmp_error = 0.0;
    for (int ij = 0; ij < m*n; ij++){
      tmp_error += fabs(Bref[idx][ij] - Bsol[idx][ij]);
    }
    if (tmp_error > error) error = tmp_error;
  }
  return error/m*n;
}
