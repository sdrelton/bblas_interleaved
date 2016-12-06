#ifndef BBLAS_S_INTL_H
#define BBLAS_S_INTL_H

#include "bblas_types.h"
#include "bblas_macros.h"

/*
 * Declarations of level 3 batched BLAS - alphabetical order
 */

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
	const int batch_count, int info);

void bblas_sgemm_intl_opt(
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
	const int batch_count, int info);

void bblas_sgemm_blkintl(
			 enum BBLAS_TRANS transA,
			 enum BBLAS_TRANS transB,
			 int M,
			 int N,
			 int K,
			 float alpha,
			 const float **Ap2p,
			 const float **Bp2p,
			 float beta, float **Cp2p,
			 float *work, int block_size,
			 int batch_count, int info);

void bblas_sgemm_blkintl_expert(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const float alpha,
	const float *arrayA,
	const float *arrayB,
	const float beta,
	float *arrayC, const int block_size,
	const int batch_count, int info);

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
	const int batch_count, int info);

void bblas_sgemm_intl_opt(
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
	const int batch_count, int info);

void bblas_sgemm_blkintl(
			 enum BBLAS_TRANS transA,
			 enum BBLAS_TRANS transB,
			 int M,
			 int N,
			 int K,
			 float alpha,
			 const float **Ap2p,
			 const float **Bp2p,
			 float beta, float **Cp2p,
			 float *work, int block_size,
			 int batch_count, int info);

void bblas_sgemm_blkintl_expert(
	const enum BBLAS_TRANS transA,
	const enum BBLAS_TRANS transB,
	const int M,
	const int N,
	const int K,
	const float alpha,
	const float *arrayA,
	const float *arrayB,
	const float beta,
	float *arrayC, const int block_size,
	const int batch_count, int info);

void bblas_strsm_intl_expert(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    float alpha,
    const float *arrayA,
    float *arrayB,
    int batch_count, int info);

void bblas_strsm_blkintl_expert(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    float alpha,
    const float * arrayA,
    float *arrayB, int block_size,
    int batch_count, int info);

void bblas_strsm_intl(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    float alpha,
    const float **Ap2p, int lda,
    float **Bp2p, int ldb,
    float *work, int batch_count, int info);

void bblas_strsm_blkintl(
		       enum BBLAS_SIDE  side,
		       enum BBLAS_UPLO uplo,
		       enum BBLAS_TRANS trans,
		       enum BBLAS_DIAG diag,
		       int m,
		       int n,
		       float alpha,
		       const float **Ap2p, int lda,
		       float **Bp2p, int  ldb, int block_size,
		       float *work, int batch_count, int info);


//Lapacke routines
void bblas_spotrf_blkintl(enum BBLAS_UPLO uplo, int n,
				float **Ap2p, int lda,
				int block_size, float *work,
				int batch_count, int info);

void bblas_spotrf_blkintl_expert(enum BBLAS_UPLO uplo, int n,
				 float *arrayAblk, int lda,
				 int block_size,
				 int batch_count, int info);

void bblas_spotrf_intl(enum BBLAS_UPLO uplo, int n,
		       float **Ap2p, int lda,
		       float *work, int batch_count, int info);

void bblas_spotrf_intl_expert(enum BBLAS_UPLO uplo, int n,
			      float *arrayA, int batch_count, int info);


void bblas_sposv_intl_expert(enum BBLAS_UPLO uplo, int m, int n,
                             float *arrayA, float *arrayB,
                             int batch_count, int info);

void bblas_sposv_intl(enum BBLAS_UPLO uplo,
                      int m, int n,
                      float **Ap2p, int lda,
                      float **Bp2p, int ldb,
                      float *work, int batch_count, int info);

void bblas_sposv_blkintl_expert(enum BBLAS_UPLO uplo,
                               int m, int n,
                               float *arrayAblk,
                               float *arrayBlk, int block_size,
                               int batch_count, int info);

void bblas_sposv_blkintl(enum BBLAS_UPLO uplo,
                        int m, int n,
                        float **Ap2p, int lda,
                        float **Bp2p, int  ldb,
                        int block_size, float *work,
                        int batch_count, int info);

//Lapacke routines
void bblas_spotrf_blkintl(enum BBLAS_UPLO uplo, int n,
				float **Ap2p, int lda,
				int block_size, float *work,
				int batch_count, int info);

void bblas_spotrf_blkintl_expert(enum BBLAS_UPLO uplo, int n,
				 float *arrayAblk, int lda,
				 int block_size,
				 int batch_count, int info);

void bblas_spotrf_intl(enum BBLAS_UPLO uplo, int n,
		       float **Ap2p, int lda,
		       float *work, int batch_count, int info);

void bblas_spotrf_intl_expert(enum BBLAS_UPLO uplo, int n,
			      float *arrayA, int batch_count, int info);


void bblas_sposv_intl_expert(enum BBLAS_UPLO uplo, int m, int n,
                             float *arrayA, float *arrayB,
                             int batch_count, int info);

void bblas_sposv_intl(enum BBLAS_UPLO uplo,
                      int m, int n,
                      float **Ap2p, int lda,
                      float **Bp2p, int ldb,
                      float *work, int batch_count, int info);

void bblas_sposv_blkintl_expert(enum BBLAS_UPLO uplo,
                               int m, int n,
                               float *arrayAblk,
                               float *arrayBlk, int block_size,
                               int batch_count, int info);

void bblas_sposv_blkintl(enum BBLAS_UPLO uplo,
                        int m, int n,
                        float **Ap2p, int lda,
                        float **Bp2p, int  ldb,
                        int block_size, float *work,
                        int batch_count, int info);


// Annexe routines for conversions and norm computation
void memcpy_sbptp2ptp(float **Bp2p, float **Bref, int m, int n, int batch_count);
void memcpy_sbptp2intl(float *arrayB, float **Bp2p, int m, int n, int batch_count);
void memcpy_sbptp2blkintl(float *arrayBblk, float **Bp2p, int m, int n, int block_size, int batch_count);
void memcpy_sbintl2ptp(float **Bp2p, float *arrayB, int m, int n, int batch_count);
void memcpy_sbblkintl2ptp(float **Bp2p, float *arrayBblk, int m, int n, int block_size, int batch_count);
void memcpy_saptp2intl(float *arrayA, float **Ap2p, int m, int batch_count);
void memcpy_saptp2blkintl(float *arrayAblk, float **Ap2p, int m, int block_size, int batch_count);
void memcpy_saintl2ptp(float **Ap2p, float *arrayA, int m, int batch_count);
void memcpy_sablkintl2ptp(float **Ap2p, float *arrayAblk, int m, int block_size, int batch_count);
float get_serror(float **Bref, float **Bsol, int m, int n, int batch_count);

#endif // BBLAS_S_INTL_H
