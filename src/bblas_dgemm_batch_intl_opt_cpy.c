#include "bblas_interleaved.h"
#include <omp.h>

// Assumes interleaved in column major order

void bblas_dgemm_batch_intl_opt(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const double alpha,
	const double *__restrict__ arrayA, const int strideA,
	const double *__restrict__ arrayB, const int strideB,
	const double beta,
	double *__restrict__ arrayC, const int strideC,
	const int batch_count, int info)
{
	// A: no transpose
	// B: no transpose
	for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					double *startA = arrayA + k*strideA*M + i*strideA;
					double *startB = arrayB + j*strideB*K + k*strideB;
					double *startC = arrayC + j*strideC*M + i*strideC;
                    //#pragma omp parallel for
					for (int idx = 0; idx < batch_count; idx++)
					{
					    if (k == 0)
						{
					        *(startC + idx) *= beta;
				        }
						*(startC + idx) += alpha* (*(startA + idx) * *(startB + idx));
   				    }
				}
			}
		}
	info = BBLAS_SUCCESS;
}
