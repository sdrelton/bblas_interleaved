#include "bblas_interleaved.h"
#include<stdio.h>
#include <math.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_dpotrf_blkintl_expert(enum BBLAS_UPLO uplo, int n,
                                 double *arrayAblk, int lda,
                                 int block_size, int batch_count, int info)
{
    // Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;

    //Block interleave variables
    int numblocks = batch_count / block_size;
    int remainder = 0;
    if (batch_count % block_size != 0)
    {
        numblocks += 1;
        remainder = (batch_count % block_size);
    }
  
    if (uplo == BblasLower) {
    
        #pragma omp parallel for
        for (int blkidx = 0; blkidx < numblocks; blkidx++) {
            int startposA = blkidx * block_size * n*(n+1)/2;
            // Remainder
            if ((blkidx == numblocks-1) && (remainder != 0)) {
                //Compute A(j,j) =  sqrt(A(j,j) -sum(A(j,i)*A(j,i))
                for (int j = 0; j < n; j++) {
                    int Ajj = (n*(n+1)/2)*blkidx*block_size + ((2*n-j-1)*j/2 + j)*block_size;
                    for (int i = 0; i < j; i++) {
                        int Aji = (n*(n+1)/2)*blkidx*block_size + ((2*n-i-1)*i/2 + j)*block_size;
                        #pragma ivdep
                        #pragma simd
                        for (int idx = 0; idx < remainder; idx++) {
                            arrayAblk[Ajj + idx] -= arrayAblk[Aji +idx]*arrayAblk[Aji + idx];
                        }
                    }
                    #pragma ivdep
                    #pragma simd  
                    for (int idx = 0; idx < remainder; idx++) {
                        arrayAblk[Ajj +idx ] = sqrt(arrayAblk[Ajj +idx ]);
                    }
                    //Update
                    for (int i = j+1; i < n; i++) {
                        int Aij = startposA + ((2*n-j-1)*j/2 +i )*block_size;
                        for (int k = 0; k <j; k++) {
                            int Ajk = startposA + ((2*n-k-1)*k/2 + j)*block_size;
                            int Aik = startposA + ((2*n-k-1)*k/2 + i)*block_size;
                            #pragma ivdep
                            #pragma simd
                            for (int idx = 0; idx < remainder; idx++) {
                                arrayAblk[Aij + idx] -= arrayAblk[Aik + idx]*arrayAblk[Ajk +idx];

                            }
                        }
                        #pragma vector aligned
                        for (int idx = 0; idx < remainder; idx++) {
                            arrayAblk[Aij +idx] = arrayAblk[Aij +idx]/arrayAblk[Ajj +idx];
                        }
                    }
                }
            } else {  
                //Compute A(j,j) =  sqrt(A(j,j) -sum(A(j,i)*A(j,i))
                for (int j = 0; j < n; j++) {
                    int Ajj = (n*(n+1)/2)*blkidx*block_size + ((2*n-j-1)*j/2 + j)*block_size;
                    for (int i = 0; i < j; i++) {
                        int Aji = (n*(n+1)/2)*blkidx*block_size + ((2*n-i-1)*i/2 + j)*block_size;
                        #pragma ivdep
                        #pragma simd
                        for (int idx = 0; idx < block_size; idx++) {
                            arrayAblk[Ajj + idx] -= arrayAblk[Aji +idx]*arrayAblk[Aji + idx];
                        }
                    }
                    #pragma ivdep
                    #pragma simd
                    for (int idx = 0; idx < block_size; idx++) {
                        arrayAblk[Ajj +idx ] = sqrt(arrayAblk[Ajj +idx ]);
                    }
                    //Update
                    for (int i = j+1; i < n; i++) {
                        int Aij = startposA + ((2*n-j-1)*j/2 +i )*block_size;
                        for (int k = 0; k <j; k++) {
                            int Ajk = startposA + ((2*n-k-1)*k/2 + j)*block_size;
                            int Aik = startposA + ((2*n-k-1)*k/2 + i)*block_size;
                            #pragma vector aligned
                            #pragma ivdep
                            #pragma simd
                            for (int idx = 0; idx < block_size; idx++) {
                                arrayAblk[Aij + idx] -= arrayAblk[Aik + idx]*arrayAblk[Ajk +idx];

                            }
                        }
                        #pragma ivdep
                        #pragma simd
                        for (int idx = 0; idx < block_size; idx++) {
                            arrayAblk[Aij +idx] = arrayAblk[Aij +idx]/arrayAblk[Ajj +idx];
                        }
                    }
                }
            }
        }
        info = BBLAS_SUCCESS;
    }
}
  
#undef COMPLEX
