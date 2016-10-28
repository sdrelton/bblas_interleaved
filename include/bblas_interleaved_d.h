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

void bblas_dgemm_batch_blkintl(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const double alpha,
	const double *arrayA,
	const double *arrayB,
	const double beta,
	double *arrayC, const int block_size,
	const int batch_count, int info);

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
    int batch_count, int info);
#endif // BBLAS_D_INTL_H
