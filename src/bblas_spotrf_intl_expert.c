#include "bblas_interleaved.h"
#include<stdio.h>
#include <math.h>
#define COMPLEX

// Assumes interleaved in column major order

void bblas_spotrf_intl_expert(enum BBLAS_UPLO uplo, int n,
                              float *arrayA, int batch_count, int info)
{
    // Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j-1)*j/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;

    int lda = n;
    if (uplo == BblasLower) {
        //Compute A(j,j) =  sqrt(A(j,j) -sum(A(j,i)*A(j,i))
        for (int j = 0; j < n; j++) {
            int Ajj = ((2*n-j-1)*j/2 + j)*batch_count;
            for (int i = 0; i < j; i++) {
                int Aji = ((2*n-i-1)*i/2 + j)*batch_count;
                #pragma omp  parallel for simd
                for (int idx = 0; idx < batch_count; idx++) {
                    arrayA[Ajj + idx] -= arrayA[Aji +idx]*arrayA[Aji + idx];
                }
            }
            #pragma omp  parallel for simd
            for (int idx = 0; idx < batch_count; idx++) {
                arrayA[Ajj +idx ] = sqrt(arrayA[Ajj +idx ]);
            }
            //Update
            for (int i = j+1; i < n; i++) {
                int Aij =  ((2*n-j-1)*j/2 +i )*batch_count;
                for (int k = 0; k <j; k++) {
                    int Ajk = ((2*n-k-1)*k/2 + j)*batch_count;
                    int Aik = ((2*n-k-1)*k/2 + i)*batch_count;
                    #pragma omp  parallel for simd
                    for (int idx = 0; idx < batch_count; idx++) {
                        arrayA[Aij + idx] -= arrayA[Aik + idx]*arrayA[Ajk +idx];
                    }
                }
                #pragma omp  parallel for simd
                for (int idx = 0; idx < batch_count; idx++) {
                    arrayA[Aij +idx] = arrayA[Aij +idx]/arrayA[Ajj +idx];
                }
            }
        }
    }
}
