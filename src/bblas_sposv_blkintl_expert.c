#include "bblas_interleaved.h"
#include<stdio.h>
# include <math.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_sposv_blkintl_expert(enum BBLAS_UPLO uplo,
                               int m, int n,
                               float *arrayAblk,
                               float *arrayBblk, int block_size,
                               int batch_count, int info) {
    // Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;
    
    int offset;
    float alpha = 1.0;
    int numblocks = batch_count / block_size;
    int remainder = 0;
    if (batch_count % block_size != 0) {
        numblocks += 1;
        remainder = (batch_count % block_size);
    }

    if (uplo != BblasLower) {
        printf("Configuration not implemented yet\n");
        return;
    }
    
    #pragma omp parallel for
    for (int blkidx = 0; blkidx < numblocks; blkidx++) {
        int startposA = blkidx * block_size * m*(m+1)/2;
        int startposB = blkidx * block_size * m*n;
        // Remainder
        if ((blkidx == numblocks-1) && (remainder != 0)) {
            //==============================================
            //Compute the Cholesky factorization A = L*L'
            //==============================================
            
            //Compute A(j,j) =  sqrt(A(j,j) -sum(A(j,i)*A(j,i))
            for (int j = 0; j < m; j++) {
                int Ajj = (m*(m+1)/2)*blkidx*block_size + ((2*m-j-1)*j/2 + j)*block_size;
                for (int i = 0; i < j; i++) {
                    int Aji = (m*(m+1)/2)*blkidx*block_size + ((2*m-i-1)*i/2 + j)*block_size;
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
                for (int i = j+1; i < m; i++) {
                    int Aij = startposA + ((2*m-j-1)*j/2 +i )*block_size;
                    for (int k = 0; k <j; k++) {
                        int Ajk = startposA + ((2*m-k-1)*k/2 + j)*block_size;
                        int Aik = startposA + ((2*m-k-1)*k/2 + i)*block_size;
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
            //========================================
            //Solve U*X = B, overwriting B with X.
            //========================================
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < m; k++) {
                    int startB = m*n*blkidx*block_size + (j*m + k)*block_size;
                    int Akk = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 + k)*block_size;
                    offset = k*block_size;		    
                    for (int i = k; i < m; i++) {
                        int Bij = m*n*blkidx*block_size + (j*m +i)*block_size ;
                        int Aik = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 +i )*block_size;
                        #pragma ivdep
                        #pragma simd
                        for (int idx = 0; idx < remainder; idx++) {
                            if (k == 0 ) arrayBblk[Bij + idx] *= alpha; // alpha B
                            if (i == k ){
                                arrayBblk[startB + idx] /= arrayAblk[Akk + idx];
                                continue; 
                            }
                            arrayBblk[Bij + idx] -=arrayBblk[startB + idx]*arrayAblk[Aik + idx];
                        }
                    }
                }
            }
            //=====================================
            //Solve U'*X = B, overwriting B with X.
            //====================================
            for (int j = 0; j < n; j++) {
                for (int i = m-1; i >= 0; i--) {
                    int Bij = m*n*blkidx*block_size + (j*m + i)*block_size;
                    int Aii = (m*(m+1)/2)*blkidx*block_size + ((2*m-i-1)*i/2 + i)*block_size;
                    for (int k = i; k <= m; k++) {
                        int Bkj = m*n*blkidx*block_size + (j*m +k)*block_size ;
                        int Aki = (m*(m+1)/2)*blkidx*block_size + ((2*m-i-1)*i/2 +k )*block_size;
                        #pragma ivdep
                        #pragma simd
                        for (int idx = 0; idx < remainder; idx++) {
                            if (k == i ) { arrayBblk[Bij + idx] *= alpha; // alpha B
                                continue;
                            }
                            if (k == m ){
                                arrayBblk[Bij + idx] /= arrayAblk[Aii + idx];
                                continue; 
                            }
                            arrayBblk[Bij + idx] -=arrayBblk[Bkj + idx]*arrayAblk[Aki + idx];
                        }
                    }
                }
            }
        } else {  
            //==============================================
            //Compute the Cholesky factorization A = L*L'
            //==============================================

            //Compute A(j,j) =  sqrt(A(j,j) -sum(A(j,i)*A(j,i))
            for (int j = 0; j < m; j++) {
                int Ajj = (m*(m+1)/2)*blkidx*block_size + ((2*m-j-1)*j/2 + j)*block_size;
                for (int i = 0; i < j; i++) {
                    int Aji = (m*(m+1)/2)*blkidx*block_size + ((2*m-i-1)*i/2 + j)*block_size;
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
                for (int i = j+1; i < m; i++) {
                    int Aij = startposA + ((2*m-j-1)*j/2 +i )*block_size;
                    for (int k = 0; k <j; k++) {
                        int Ajk = startposA + ((2*m-k-1)*k/2 + j)*block_size;
                        int Aik = startposA + ((2*m-k-1)*k/2 + i)*block_size;
                        #pragma ivdep
                        #pragma simd
                        for (int idx = 0; idx < block_size; idx++) {
                            arrayAblk[Aij + idx] -= arrayAblk[Aik + idx]*arrayAblk[Ajk +idx];
                                
                        }
                    }
                    #pragma vector aligned
                    for (int idx = 0; idx < block_size; idx++) {
                        arrayAblk[Aij +idx] = arrayAblk[Aij +idx]/arrayAblk[Ajj +idx];
                    }
                }
            }
            //========================================
            //Solve U*X = B, overwriting B with X.
            //========================================
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < m; k++) {
                    int startB = m*n*blkidx*block_size + (j*m + k)*block_size;
                    int Akk = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 + k)*block_size;
                    offset = k*block_size;		    
                    for (int i = k; i < m; i++) {
                        int Bij = m*n*blkidx*block_size + (j*m +i)*block_size ;
                        int Aik = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 +i )*block_size;
                        #pragma ivdep
                        #pragma simd
                        for (int idx = 0; idx < block_size; idx++) {
                            if (k == 0 ) arrayBblk[Bij + idx] *= alpha; // alpha B
                            if (i == k ){
                                arrayBblk[startB + idx] /= arrayAblk[Akk + idx];
                                continue; 
                            }
                            arrayBblk[Bij + idx] -=arrayBblk[startB + idx]*arrayAblk[Aik + idx];
                        }
                    }
                }
            }
            // TRSM CblasTrans
            for (int j = 0; j < n; j++) {
                for (int i = m-1; i >= 0; i--) {
                    int Bij = m*n*blkidx*block_size + (j*m + i)*block_size;
                    int Aii = (m*(m+1)/2)*blkidx*block_size + ((2*m-i-1)*i/2 + i)*block_size;
                    for (int k = i; k <= m; k++) {
                        int Bkj = m*n*blkidx*block_size + (j*m +k)*block_size ;
                        int Aki = (m*(m+1)/2)*blkidx*block_size + ((2*m-i-1)*i/2 +k )*block_size;
                        #pragma ivdep
                        #pragma simd
                        for (int idx = 0; idx < block_size; idx++) {
                            if (k == i ) { arrayBblk[Bij + idx] *= alpha; // alpha B
                                continue;
                            }
                            if (k == m ){
                                arrayBblk[Bij + idx] /= arrayAblk[Aii + idx];
                                continue; 
                            }
                            arrayBblk[Bij + idx] -=arrayBblk[Bkj + idx]*arrayAblk[Aki + idx];
                        }
                    }
                }
            }
        }
    }
}
