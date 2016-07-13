#include <complex.h>
#include "bblas_interleaved.h"

#define COMPLEX

// Assumes interleaved in column major order

void bblas_zgemm_batch_intl(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const BBLAS_Complex64_t alpha,
	const BBLAS_Complex64_t *arrayA, const int strideA,
	const BBLAS_Complex64_t *arrayB, const int strideB,
	const BBLAS_Complex64_t beta,
	BBLAS_Complex64_t *arrayC, const int strideC,
	const int batch_count, int info)
{
	// Error checks go here

	// Note: arrayA(i,k,idx) = arrayA[k*strideA*M + i*strideA + idx]

	// Multiply C by beta
	for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] *=
							beta;
					}
				}
			}
		}


	if (transA == BblasNoTrans && transB == BblasNoTrans)
	{
		// A: no transpose
		// B: no transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							arrayA[k*strideA*M + i*strideA + idx]*
							arrayB[j*strideB*K + k*strideB + idx];
					}
				}
			}
		}
	}
	else if (transA == BblasTrans && transB == BblasNoTrans)
	{
		// A: transpose
		// B: no transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							arrayA[i*strideA*K + k*strideA + idx]*
							arrayB[j*strideB*K + k*strideB + idx];
					}
				}
			}
		}
	}
	else if (transA == BblasConjTrans && transB == BblasNoTrans)
	{
		// A: conjugate transpose
		// B: no transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							conj(arrayA[i*strideA*K + k*strideA + idx])*
							arrayB[j*strideB*K + k*strideB + idx];
					}
				}
			}
		}
	}
	else if (transA == BblasNoTrans && transB == BblasTrans)
	{
		// A: no transpose
		// B: transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							arrayA[k*strideA*M + i*strideA + idx]*
							arrayB[k*strideB*N + j*strideB + idx];
					}
				}
			}
		}
	}
	else if (transA == BblasTrans && transB == BblasTrans)
	{
		// A: transpose
		// B: transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							arrayA[i*strideA*K + k*strideA + idx]*
							arrayB[k*strideB*N + j*strideB + idx];
					}
				}
			}
		}
	}
	else if (transA == BblasConjTrans && transB == BblasTrans)
	{
		// A: conjugate traspose
		// B: transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							conj(arrayA[i*strideA*K + k*strideA + idx])*
							arrayB[k*strideB*N + j*strideB + idx];
					}
				}
			}
		}
	}
	else if (transA == BblasNoTrans && transB == BblasConjTrans)
	{
		// A: no transpose
		// B: conjugate transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							arrayA[k*strideA*M + i*strideA + idx]*
							conj(arrayB[k*strideB*N + j*strideB + idx]);
					}
				}
			}
		}
	}
	else if (transA == BblasTrans && transB == BblasConjTrans)
	{
		// A: transpose
		// B: conjugate transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							arrayA[i*strideA*K + k*strideA + idx]*
							conj(arrayB[k*strideB*N + j*strideB + idx]);
					}
				}
			}
		}
	}
	else if (transA == BblasConjTrans && transB == BblasConjTrans)
	{
		// A: conjugate transpose
		// B: conjugate transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					for (int idx = 0; idx < batch_count; idx++)
					{
						arrayC[j*strideC*M + i*strideC + idx] +=
							alpha*
							conj(arrayA[i*strideA*K + k*strideA + idx])*
							conj(arrayB[k*strideB*N + j*strideB + idx]);
					}
				}
			}
		}
	}
	info = BBLAS_SUCCESS;
}

#undef COMPLEX
