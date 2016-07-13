#ifndef BBLAS_D_INTL_H
#define BBLAS_D_INTL_H

#include "bblas_types.h"
#include "bblas_macros.h"

/*
 * Declarations of level 3 batched BLAS - alphabetical order
 */
void bblas_dgemm_batch_intl(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const double alpha,
	const double *arrayA, const int strideA,
	const double *arrayB, const int strideB,
	const double beta,
	double *arrayC, const int strideC,
	const int batch_count, int info);

void bblas_dgemm_batch_intl_opt(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const double alpha,
	const double *arrayA, const int strideA,
	const double *arrayB, const int strideB,
	const double beta,
	double *arrayC, const int strideC,
	const int batch_count, int info);

#endif // BBLAS_Z_INTL_H