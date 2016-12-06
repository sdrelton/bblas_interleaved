#include "bblas_interleaved.h"
#include<stdio.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_strsm_blkintl(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    float alpha,
    const float **Ap2p, int lda,
    float **Bp2p, int  ldb, int block_size,
    float *work, int batch_count, int info) {

    // Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;
    
    int offset;
    
    //Block interleave variables
    float *arrayAblk = work;
    float *arrayBblk = (work+m*m*batch_count);

    int numblocks = batch_count / block_size;
    int remainder = 0;
    if (batch_count % block_size != 0) {
        numblocks += 1;
        remainder = (batch_count % block_size);
    }

    if ((side != BblasLeft)
        || (uplo != BblasLower)
        || (diag != BblasNonUnit)) {
        printf("Configuration not implemented yet\n");
        return;
    }
    if (trans == BblasNoTrans) {
        #pragma omp parallel for
        for (int blkidx = 0; blkidx < numblocks; blkidx++) {
            int startposA = blkidx * block_size * m*(m+1)/2;
            int startposB = blkidx * block_size * m*n;
            // Remainder
            if ((blkidx == numblocks-1) && (remainder != 0)) {
                //Convert Ap2p -> A block interleave
                for (int j = 0; j < m; j++)
                    for (int i = j; i < m; i++) {
                        offset = startposA + (j*(2*m-j-1)/2 + i)*block_size;
                        #pragma ivdep
                        for (int idx = 0; idx < remainder; idx++) {
                            arrayAblk[offset + idx] = Ap2p[blkidx * block_size + idx][j*lda+i];
                        }
                    }
                //Convert Bp2p -> B block interleave
                for (int pos = 0; pos < m*n; pos++) {
                    offset = startposB + pos*block_size;
                    #pragma ivdep	      
                    for (int idx = 0; idx < remainder; idx++) {
                        arrayBblk[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
                    }
                }
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
                // Convert Bblk -> Bp2p
                for (int pos = 0; pos < m*n; pos++) {
                    offset = startposB + pos*block_size;
                    #pragma ivdep
                    for (int idx = 0; idx < remainder; idx++) {
                        Bp2p[blkidx * block_size + idx][pos] = arrayBblk[offset + idx];
                    }
                }        
            } else {  
                //Convert Ap2p -> A block interleave
                for (int j = 0; j < m; j++) 
                    for (int i = j; i < m; i++ ) {
                        offset = startposA + (j*(2*m-j-1)/2 + i)*block_size;
                        #pragma ivdep	      
                        for (int idx = 0; idx < block_size; idx++) {
                            arrayAblk[offset + idx] = Ap2p[blkidx * block_size + idx][j*lda+i];
                        }
                    }
                //Convert Bp2p -> B block interleave
                for (int pos = 0; pos < m*n; pos++) {
                    offset = startposB + pos*block_size;
                    #pragma ivdep
                    for (int idx = 0; idx < block_size; idx++) {
                        arrayBblk[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
                    }
                }
                //Compute trsm
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < m; k++) {
                        int startB = m*n*blkidx*block_size + (j*m + k)*block_size;
                        int Akk = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 + k)*block_size;
                        offset = k*block_size;		    
                        for (int i = k; i < m; i++) {
                            int Bij = m*n*blkidx*block_size + (j*m +i)*block_size ;
                            int Aik = (m*(m+1)/2)*blkidx*block_size + ((2*m-k-1)*k/2 +i )*block_size;
                            //#pragma ivdep
                            //#pragma simd
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
                // Convert Bblk -> Bp2p
                for (int pos = 0; pos < m*n; pos++) {
                    int offset = startposB + pos*block_size;
                    //#pragma ivdep
                    for (int idx = 0; idx < block_size; idx++) {
                        Bp2p[blkidx * block_size + idx][pos] = arrayBblk[offset + idx];
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for
        for (int blkidx = 0; blkidx < numblocks; blkidx++) {
            int startposA = blkidx * block_size * m*(m+1)/2;
            int startposB = blkidx * block_size * m*n;
            // Remainder
            if ((blkidx == numblocks-1) && (remainder != 0)) {
                //Convert Ap2p -> A block interleave
                for (int j = 0; j < m; j++)
                    for (int i = j; i < m; i++) {
                        offset = startposA + (j*(2*m-j-1)/2 + i)*block_size;
                        #pragma ivdep
                        for (int idx = 0; idx < remainder; idx++) {
                            arrayAblk[offset + idx] = Ap2p[blkidx * block_size + idx][j*lda+i];
                        }
                    }
                //Convert Bp2p -> B block interleave
                for (int pos = 0; pos < m*n; pos++) {
                    offset = startposB + pos*block_size;
                    #pragma ivdep	      
                    for (int idx = 0; idx < remainder; idx++) {
                        arrayBblk[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
                    }
                }
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
                // Convert Bblk -> Bp2p
                for (int pos = 0; pos < m*n; pos++) {
                    offset = startposB + pos*block_size;
                    #pragma ivdep
                    for (int idx = 0; idx < remainder; idx++) {
                        Bp2p[blkidx * block_size + idx][pos] = arrayBblk[offset + idx];
                    }
                }        
            } else {
                //Convert Ap2p -> A block interleave
                for (int j = 0; j < m; j++)
                    for (int i = j; i < m; i++) {
                        offset = startposA + (j*(2*m-j-1)/2 + i)*block_size;
                        #pragma ivdep
                        for (int idx = 0; idx < block_size; idx++) {
                            arrayAblk[offset + idx] = Ap2p[blkidx * block_size + idx][j*lda+i];
                        }
                    }
                //Convert Bp2p -> B block interleave
                for (int pos = 0; pos < m*n; pos++) {
                    offset = startposB + pos*block_size;
                    #pragma ivdep	      
                    for (int idx = 0; idx < block_size; idx++) {
                        arrayBblk[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
                    }
                }
                
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
                // Convert Bblk -> Bp2p
                for (int pos = 0; pos < m*n; pos++) {
                    offset = startposB + pos*block_size;
                    #pragma ivdep
                    for (int idx = 0; idx < block_size; idx++) {
                        Bp2p[blkidx * block_size + idx][pos] = arrayBblk[offset + idx];
                    }
                }        
            }
        }
    }
    info = BBLAS_SUCCESS;
}
#undef COMPLEX
    
