#include "bblas_interleaved.h"

#define COMPLEX

// Assumes interleaved in column major order

void bblas_sgemm_intl(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const float alpha,
	const float *arrayA, const int strideA,
	const float *arrayB, const int strideB,
	const float beta,
	float *arrayC, const int strideC,
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
	else if (transA == BblasTrans && transB == BblasNoTrans)
	{
		// A: ugate transpose
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
							(arrayA[i*strideA*K + k*strideA + idx])*
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
	else if (transA == BblasTrans && transB == BblasTrans)
	{
		// A: ugate traspose
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
							(arrayA[i*strideA*K + k*strideA + idx])*
							arrayB[k*strideB*N + j*strideB + idx];
					}
				}
			}
		}
	}
	else if (transA == BblasNoTrans && transB == BblasTrans)
	{
		// A: no transpose
		// B: ugate transpose
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
							(arrayB[k*strideB*N + j*strideB + idx]);
					}
				}
			}
		}
	}
	else if (transA == BblasTrans && transB == BblasTrans)
	{
		// A: transpose
		// B: ugate transpose
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
							(arrayB[k*strideB*N + j*strideB + idx]);
					}
				}
			}
		}
	}
	else if (transA == BblasTrans && transB == BblasTrans)
	{
		// A: ugate transpose
		// B: ugate transpose
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
							(arrayA[i*strideA*K + k*strideA + idx])*
							(arrayB[k*strideB*N + j*strideB + idx]);
					}
				}
			}
		}
	}
	info = BBLAS_SUCCESS;
}

#undef COMPLEX
