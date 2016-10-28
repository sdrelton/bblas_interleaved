#include "bblas_interleaved.h"

#define COMPLEX

// Assumes interleaved in column major order

void bblas_dtrsm_batch_intl(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    double alpha,
	const double *arrayA, int strideA,
    double *arrayB, int strideB,
    int batch_count, int info)
{
	// Error checks go here
    // if UPLO = `L', aij is stored in A( i+(2*m-j)*(j-1)/2) for $j \leq i$.
    //if UPLO = `U', aij is stored in A(i+j*(j-1)/2) for $i \leq j$;
	// Note: arrayB(i,k,idx) = arrayB[k*strideA*M + i*strideA + idx]

	if ((side == BblasLeft)
        && (uplo == BblasLower)
        && (diag == BblasNonUnit)
        && (trans == BblasNoTrans) ) {

        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k++)
            {
                int startB = (j*m + k)*strideB;
                for (int i = k; i < m; i++)
                {
                    #pragma omp parallel for
                    for (int idx = 0; idx < batch_count; idx++)
                    {
                        if (i == 0 )
						{
                            if (arrayB[startB + idx] != 0 ) {
                                arrayB[startB + idx] *= alpha;
                                // arrayB[startB + idx] /= arrayA[((2*m-k)*(k-1)/2 + k)*strideA + idx];
                                arrayB[startB + idx] /= arrayA[(k*m + k)*strideA + idx]; 
                            }
				        }
                        // arrayB[(j*m + i)*strideB + idx] -=  arrayB[startB + idx]*arrayA[((2*m-k)*(k-1)/2 + i)*strideA + idx];
                        arrayB[(j*m + i)*strideB + idx] -=  arrayB[startB + idx]*arrayA[(m*k + i)*strideA + idx];
                    }
                }
            }
        }
    }
    info = BBLAS_SUCCESS;
}

#undef COMPLEX
