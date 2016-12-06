#include "bblas_interleaved.h"
#include <omp.h>

// Assumes interleaved in column major order

void bblas_sgemm_blkintl_expert(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const float alpha,
	const float *__restrict__ arrayA,
	const float *__restrict__ arrayB,
	const float beta,
	float * arrayC, const int block_size,
	const int batch_count, int info)
{
	// Error checks go here
	int numblocks = batch_count / block_size;
	int remainder = 0;
	if (batch_count % block_size != 0)
	{
		numblocks += 1;
		remainder = (batch_count % block_size);
	}

	if (transA == BblasNoTrans && transB == BblasNoTrans)
	{
		// Try vectorized version
		// A: no transpose
		// B: no transpose
		# pragma omp parallel for
		for (int blkidx = 0; blkidx < numblocks; blkidx++)
		{
			// Remainder
			if ((blkidx == numblocks-1) && (remainder != 0))
			{
				for (int i = 0; i < M; i++)
				{
					for (int j = 0; j < N; j++)
					{
						for (int k = 0; k < K; k++)
						{
							int startA = M*K*blkidx*block_size + k*block_size*M + i*block_size;
							int startB = K*N*blkidx*block_size + j*block_size*K + k*block_size;
							int startC = M*N*blkidx*block_size + j*block_size*M + i*block_size;
							#pragma vector aligned
							for (int idx = 0; idx < remainder; idx++)
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
			} // end if
			else
			{
				for (int i = 0; i < M; i++)
				{
					for (int j = 0; j < N; j++)
					{
						for (int k = 0; k < K; k++)
						{
							int startA = M*K*blkidx*block_size + k*block_size*M + i*block_size;
							int startB = K*N*blkidx*block_size + j*block_size*K + k*block_size;
							int startC = M*N*blkidx*block_size + j*block_size*M + i*block_size;
							#pragma vector aligned
							for (int idx = 0; idx < block_size; idx++)
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
			} // end else
		} // end for
		info = BBLAS_SUCCESS;
	}
}
