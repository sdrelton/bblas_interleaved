#include "bblas_interleaved.h"
#include<stdio.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_strsm_intl(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    float alpha,
    const float **Ap2p, int lda,
    float **Bp2p, int ldb,
    float *work, int batch_count, int info)
{
    // Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;
    // Note: arrayB(i,k,idx) = arrayB[k*strideA*M + i*strideA + idx]
  

    float *arrayA = work;
    float *arrayB = (work + m*m*batch_count);
    int strideB = batch_count;
    int strideA = batch_count;

    // Convert Ap2p to interleaved layout
    memcpy_saptp2intl(arrayA, Ap2p, m, batch_count);
  
    // Convert Bp2p to interleaved layout
    memcpy_sbptp2intl(arrayB, Bp2p, m, n, batch_count);
  
    if ((side != BblasLeft)
        || (uplo != BblasLower)
        || (diag != BblasNonUnit)) {
        printf("Configuration not implemented yet\n");
        return;
    }
  
    if (trans == BblasNoTrans) {
    
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k++)
            {
                int Bkj = (j*m + k)*strideB;
                int Akk = ((2*m-k-1)*k/2 + k)*strideA;
                for (int i = k; i < m; i++) {
                    int Bij = (j*m+i)*strideB;
                    int Aik = ((2*m-k-1)*k/2 + i)*strideA;
                    #pragma omp parallel for simd
                    for (int idx = 0; idx < batch_count; idx++)
                    {
                        if (k == 0 ) arrayB[ Bij + idx] *= alpha; // alpha B
                        if (i == k) {
                            arrayB[Bkj + idx] /= arrayA[Akk + idx];
                            continue;
                        } 
                        arrayB[Bij + idx] -=  arrayB[Bkj + idx]*arrayA[ Aik + idx];
                    }
                }
            }
        }
    } else {
        for (int j = 0;  j < n; j++) {
            for (int i = m-1; i >= 0; i--) {
                int Bij = (j*m + i)*strideB;
                int Aii = ((2*m-i-1)*i/2 + i)*strideA;
                for (int k = i; k <= m; k++) {
                    int Bkj = (j*m+k)*strideB;
                    int Aki = ((2*m-i-1)*i/2 + k)*strideA;
                    #pragma omp parallel for simd
                    for (int idx = 0; idx < batch_count; idx++)  {
                        if (k == i) {
                            arrayB[Bij + idx] *= alpha; // alpha B
                            continue;
                        }
                        if  (k == m) {
                            arrayB[Bij + idx] /= arrayA[Aii + idx];
                            continue;
                        }
                        arrayB[Bij + idx] -=  arrayB[Bkj + idx]*arrayA[ Aki + idx];
                    }
                }
            }
        }
    }
    // convert solution back
    memcpy_sbintl2ptp(Bp2p, arrayB, m, n, batch_count);
    info = BBLAS_SUCCESS;
}

#undef COMPLEX
