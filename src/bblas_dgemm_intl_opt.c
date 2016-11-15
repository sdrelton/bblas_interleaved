#include "bblas_interleaved.h"
#include <omp.h>

// Assumes interleaved in column major order

void bblas_dgemm_intl_opt(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const double alpha,
	const double *__restrict__ arrayA, const int strideA,
	const double *__restrict__ arrayB, const int strideB,
	const double beta,
	double * arrayC, const int strideC,
	const int batch_count, int info)
{
	// Error checks go here

	// Note: arrayA(i,k,idx) = arrayA[k*strideA*M + i*strideA + idx]

	// Multiply C by beta
	/* for (int i = 0; i < M; i++) */
	/* { */
	/* 	for (int j = 0; j < N; j++) */
	/* 	{ */
	/* 		for (int k = 0; k < K; k++) */
	/* 		{ */
    /*             #pragma omp parallel for */
	/* 			for (int idx = 0; idx < batch_count; idx++) */
	/* 			{ */
	/* 				arrayC[j*strideC*M + i*strideC + idx] *= */
	/* 					beta; */
	/* 			} */
	/* 		} */
	/* 	} */
	/* } */


	if (transA == BblasNoTrans && transB == BblasNoTrans)
	{
		/* // A: no transpose */
/* 		// B: no transpose */
/* 		for (int i = 0; i < M; i++) */
/* 		{ */
/* 			for (int j = 0; j < N; j++) */
/* 			{ */
/* #pragma omp parallel for */
/* 				for (int idx = 0; idx < batch_count; idx++) */
/* 				{ */
/* 					arrayC[j*strideC*M + i*strideC + idx] *= */
/* 						beta; */
/* 					for (int k = 0; k < K; k++) */
/* 					{ */
/* 						arrayC[j*strideC*M + i*strideC + idx] += */
/* 							alpha* */
/* 							arrayA[k*strideA*M + i*strideA + idx]* */
/* 							arrayB[j*strideB*K + k*strideB + idx]; */
/* 					} */
/* 				} */
/* 			} */
/* 		} */
		// Try vectorized version
		// A: no transpose
		// B: no transpose
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < K; k++)
				{
					int startA = k*strideA*M + i*strideA;
					int startB = j*strideB*K + k*strideB;
					int startC = j*strideC*M + i*strideC;
                    #pragma omp parallel for
					for (int idx = 0; idx < batch_count; idx++)
					{
					    if (k == 0)
						{
					        arrayC[startC + idx] *= beta;
				        }
						arrayC[startC + idx] += alpha*
							(arrayA[startA + idx] * arrayB[startB + idx]);
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
				#pragma omp parallel for
				for (int idx = 0; idx < batch_count; idx++)
				{
					arrayC[j*strideC*M + i*strideC + idx] *=
						beta;
					for (int k = 0; k < K; k++)
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
                #pragma omp parallel for
				for (int idx = 0; idx < batch_count; idx++)
				{
					arrayC[j*strideC*M + i*strideC + idx] *=
						beta;
					for (int k = 0; k < K; k++)
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
                #pragma omp parallel for
				for (int idx = 0; idx < batch_count; idx++)
				{
					arrayC[j*strideC*M + i*strideC + idx] *=
						beta;
					for (int k = 0; k < K; k++)
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
				#pragma omp parallel for
				for (int idx = 0; idx < batch_count; idx++)
				{
					arrayC[j*strideC*M + i*strideC + idx] *=
						beta;
					for (int k = 0; k < K; k++)
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
                #pragma omp parallel for
				for (int idx = 0; idx < batch_count; idx++)
				{
					arrayC[j*strideC*M + i*strideC + idx] *=
						beta;
					for (int k = 0; k < K; k++)
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
                #pragma omp parallel for
				for (int idx = 0; idx < batch_count; idx++)
				{
					arrayC[j*strideC*M + i*strideC + idx] *=
						beta;
					for (int k = 0; k < K; k++)
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
                #pragma omp parallel for
				for (int idx = 0; idx < batch_count; idx++)
				{
					arrayC[j*strideC*M + i*strideC + idx] *=
						beta;
					for (int k = 0; k < K; k++)
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
                #pragma omp parallel for
				for (int idx = 0; idx < batch_count; idx++)
				{
					arrayC[j*strideC*M + i*strideC + idx] *=
						beta;
					for (int k = 0; k < K; k++)
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
