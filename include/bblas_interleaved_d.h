#ifndef BBLAS_D_INTL_H
#define BBLAS_D_INTL_H

#include "bblas_types.h"
#include "bblas_macros.h"

/*
 * Declarations of level 3 batched BLAS - alphabetical order
 */
void bblas_dgemm_intl(
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

void bblas_dgemm_intl_opt(
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

void bblas_dgemm_blkintl(
			 enum BBLAS_TRANS transA,
			 enum BBLAS_TRANS transB,
			 int M,
			 int N,
			 int K,
			 double alpha,
			 const double **Ap2p,
			 const double **Bp2p,
			 double beta, double **Cp2p,
			 double *work, int block_size,
			 int batch_count, int info);

void bblas_dgemm_blkintl_expert(
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

void bblas_dtrsm_intl_expert(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    double alpha,
    const double *arrayA,
    double *arrayB,
    int batch_count, int info);

void bblas_dtrsm_blkintl_expert(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    double alpha,
    const double * arrayA,
    double *arrayB, int block_size,
    int batch_count, int info);

void bblas_dtrsm_intl(
    enum BBLAS_SIDE  side,
    enum BBLAS_UPLO uplo,
    enum BBLAS_TRANS trans,
    enum BBLAS_DIAG diag,
    int m,
    int n,
    double alpha,
    const double **Ap2p, int lda,
    double **Bp2p, int ldb,
    double *work, int batch_count, int info);

void bblas_dtrsm_blkintl(
		       enum BBLAS_SIDE  side,
		       enum BBLAS_UPLO uplo,
		       enum BBLAS_TRANS trans,
		       enum BBLAS_DIAG diag,
		       int m,
		       int n,
		       double alpha,
		       const double **Ap2p, int lda,
		       double **Bp2p, int  ldb, int block_size,
		       double *work, int batch_count, int info);


//Lapacke routines
void bblas_dpotrf_blkintl(enum BBLAS_UPLO uplo, int n,
				double **Ap2p, int lda,
				int block_size, double *work,
				int batch_count, int info);

void bblas_dpotrf_blkintl_expert(enum BBLAS_UPLO uplo, int n,
				 double *arrayAblk, int lda,
				 int block_size,
				 int batch_count, int info);

void bblas_dpotrf_intl(enum BBLAS_UPLO uplo, int n,
		       double **Ap2p, int lda,
		       double *work, int batch_count, int info);

void bblas_dpotrf_intl_expert(enum BBLAS_UPLO uplo, int n,
			      double *arrayA, int batch_count, int info);


void bblas_dposv_intl_expert(enum BBLAS_UPLO uplo, int m, int n,
                             double *arrayA, double *arrayB,
                             int batch_count, int info);

void bblas_dposv_intl(enum BBLAS_UPLO uplo,
                      int m, int n,
                      double **Ap2p, int lda,
                      double **Bp2p, int ldb,
                      double *work, int batch_count, int info);

void bblas_dposv_blkintl_expert(enum BBLAS_UPLO uplo,
                               int m, int n,
                               double *arrayAblk,
                               double *arrayBlk, int block_size,
                               int batch_count, int info);

void bblas_dposv_blkintl(enum BBLAS_UPLO uplo,
                        int m, int n,
                        double **Ap2p, int lda,
                        double **Bp2p, int  ldb,
                        int block_size, double *work,
                        int batch_count, int info);

// Annexe routines for conversions and norm computation
void memcpy_dbptp2ptp(double **Bp2p, double **Bref, int m, int n, int batch_count);
void memcpy_dbptp2intl(double *arrayB, double **Bp2p, int m, int n, int batch_count);
void memcpy_dbptp2blkintl(double *arrayBblk, double **Bp2p, int m, int n, int block_size, int batch_count);
void memcpy_dbintl2ptp(double **Bp2p, double *arrayB, int m, int n, int batch_count);
void memcpy_dbblkintl2ptp(double **Bp2p, double *arrayBblk, int m, int n, int block_size, int batch_count);
void memcpy_daptp2intl(double *arrayA, double **Ap2p, int m, int batch_count);
void memcpy_daptp2blkintl(double *arrayAblk, double **Ap2p, int m, int block_size, int batch_count);
void memcpy_daintl2ptp(double **Ap2p, double *arrayA, int m, int batch_count);
void memcpy_dablkintl2ptp(double **Ap2p, double *arrayAblk, int m, int block_size, int batch_count);
double get_derror(double **Bref, double **Bsol, int m, int n, int batch_count);

#endif // BBLAS_D_INTL_H
