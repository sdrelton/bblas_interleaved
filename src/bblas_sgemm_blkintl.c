#include "bblas_interleaved.h"
#include <omp.h>

// Assumes interleaved in column major order

void bblas_sgemm_blkintl(
    enum BBLAS_TRANS transA,
    enum BBLAS_TRANS transB,
    int M,
    int N,
    int K,
    float alpha,
    const float **Ap2p,
    const float **Bp2p,
    float beta, float **Cp2p,
    float *work, int block_size,
    int batch_count, int info)
{
    //Local variable for conversion
    float *__restrict arrayA = work;
    float * __restrict arrayB = (arrayA + M*K*batch_count);
    float *arrayC = (arrayB + K*N*batch_count);
    int offset;
    int lda = M;
    int ldb = K;
    int ldc = M;
  
    int numblocks = batch_count / block_size;
    int remainder = 0;
    if (batch_count % block_size != 0)
    {
        numblocks += 1;
        remainder = (batch_count % block_size);
    }
  
    if (transA == BblasNoTrans && transB == BblasNoTrans)
    {
        # pragma omp parallel for
        for (int blkidx = 0; blkidx < numblocks; blkidx++) {
            int startblkA = M*K*blkidx*block_size;
            int startblkB = K*N*blkidx*block_size;
            int startblkC = M*N*blkidx*block_size;
            // Remainder
            if ((blkidx == numblocks-1) && (remainder != 0)) {
	  
                //Convert Ap2p to block interleave A
                for (int pos = 0; pos < M*K; pos++) {
                    int offset = startblkA + pos*block_size;
                    #pragma ivdep
                    #pragma simd
                    for (int idx = 0; idx < remainder; idx++) {
                        arrayA[offset + idx] = Ap2p[blkidx * block_size + idx][pos];
                    }
                }	  
                //Convert Bp2p to block interleave B
                for (int pos = 0; pos < K*N; pos++) {
                    offset = startblkB + pos*block_size;
                    #pragma ivdep
                    #pragma simd
                    for (int idx = 0; idx < remainder; idx++) {
                        arrayB[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
                    }
                }	  
                //Convert Cp2p to block interleave C
                for (int pos = 0; pos < M*N; pos++) {
                    offset = startblkC + pos*block_size;
                    #pragma ivdep
                    #pragma simd
                    for (int idx = 0; idx < remainder; idx++) {
                        arrayC[offset + idx] = Cp2p[blkidx * block_size + idx][pos];
                    }
                }	  
	  
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        for (int k = 0; k < K; k++) {
                            int startA = startblkA + k*block_size*M + i*block_size;
                            int startB = startblkB + j*block_size*K + k*block_size;
                            int startC = startblkC + j*block_size*M + i*block_size;
                            #pragma ivdep
                            #pragma simd
                            for (int idx = 0; idx < remainder; idx++) {
                                if (k == 0) {
                                    arrayC[startC + idx] *= beta;
                                }
                                arrayC[startC + idx] += alpha*
                                    (arrayA[startA + idx] * arrayB[startB + idx]);
                            }
                        }
                    }
                }
                //Convert C back to Cp2p
                for (int pos = 0; pos < M*N; pos++) {
                    offset = startblkC + pos*block_size;
                    #pragma ivdep
                    for (int idx = 0; idx < remainder; idx++) {
                        Cp2p[blkidx * block_size + idx][pos] = arrayC[offset + idx];
                    }
                }
	  
            } // end if
            else {
                //Convert Ap2p to block interleave A
                for (int pos = 0; pos < M*K; pos++) {
                    int offset = startblkA + pos*block_size;
                    #pragma ivdep
                    #pragma simd
                    for (int idx = 0; idx < block_size; idx++) {
                        arrayA[offset + idx] = Ap2p[blkidx * block_size + idx][pos];
                    }
                }	  

                //Convert Bp2p to block interleave B
                for (int pos = 0; pos < K*N; pos++) {
                    offset = startblkB + pos*block_size;
                    #pragma ivdep
                    #pragma simd
                    for (int idx = 0; idx < block_size; idx++) {
                        arrayB[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
                    }
                }	  
                //Convert Cp2p to block interleave C
                for (int pos = 0; pos < M*N; pos++) {
                    offset = startblkC + pos*block_size;
                    #pragma ivdep
                    #pragma simd
                    for (int idx = 0; idx < block_size; idx++) {
                        arrayC[offset + idx] = Cp2p[blkidx * block_size + idx][pos];
                    }
                }	  
	  
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        for (int k = 0; k < K; k++) {
                            int startA = startblkA + k*block_size*M + i*block_size;
                            int startB = startblkB + j*block_size*K + k*block_size;
                            int startC = startblkC + j*block_size*M + i*block_size;
                            #pragma ivdep
                            #pragma simd
                            for (int idx = 0; idx < block_size; idx++) {
                                if (k == 0) {
                                    arrayC[startC + idx] *= beta;
                                }
                                arrayC[startC + idx] += alpha*
                                    (arrayA[startA + idx] * arrayB[startB + idx]);
                            }
                        }
                    }
                }
                //Convert C back to Cp2p
                for (int pos = 0; pos < M*N; pos++) {
                    offset = startblkC + pos*block_size;
                    #pragma ivdep
                    #pragma simd
                    for (int idx = 0; idx < block_size; idx++) {
                        Cp2p[blkidx * block_size + idx][pos] = arrayC[offset + idx];
                    }
                }
            } // end else
        } // end for
        info = BBLAS_SUCCESS;
    }
}
