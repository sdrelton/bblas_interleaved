#ifndef BBLAS_Z_INTL_H
#define BBLAS_Z_INTL_H

#define COMPLEX

/*
 * Declarations of level 3 batched BLAS - alphabetical order
 */
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
	const int batch_count, int info);

void bblas_zgemm_batch_intl_opt(
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
	const int batch_count, int info);

#undef COMPLEX
#endif // BBLAS_Z_INTL_H
