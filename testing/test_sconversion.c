#include <stdlib.h>
#include <string.h>
#include "bblas_interleaved.h"
#include <omp.h>
#include <math.h>


void memcpy_sbptp2intl(float *arrayB, float **Bp2p, int m, int n, int batch_count){

    for (int pos = 0; pos < m*n; pos++)
    {
        int offset = pos*batch_count;
        #pragma omp parallel for
        #pragma ivdep
        for (int idx = 0; idx < batch_count; idx++)
        {
            arrayB[offset + idx] = Bp2p[idx][pos];
        }
    }
  
}
void memcpy_sbptp2blkintl(float *arrayBblk, float **Bp2p, int m, int n, int block_size, int batch_count) {

    int blocksrequired = batch_count / block_size;
    int remainder = 0;
    if (batch_count % block_size != 0)
    {
        blocksrequired += 1;
        remainder = batch_count % block_size;
    }

    #pragma omp parallel for
    #pragma ivdep
    for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
    {
        int  startpos = blkidx * block_size * m*n;
        if (blkidx == blocksrequired - 1 && remainder != 0)
        {
            // Remainders
            for (int pos = 0; pos < m*n; pos++) {
                int offset = startpos + pos*block_size;
                for (int idx = 0; idx < remainder; idx++)
                {
                    arrayBblk[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
                }
            }
        }
        else
        {
            for (int pos = 0; pos < m*n; pos++)
            {
                int offset = startpos + pos*block_size;
                for (int idx = 0; idx < block_size; idx++)
                {
                    arrayBblk[offset + idx] = Bp2p[blkidx * block_size + idx][pos];
                }
            }
        }
    }
}

void memcpy_saptp2intl(float *arrayA, float **Ap2p, int m, int batch_count) {
    int lda = m;
    for (int j = 0; j < m; j++) {
        for (int i = j; i < m; i++ ) {
            int offset = (j*(2*m-j-1)/2 + i)*batch_count;
            #pragma omp parallel for
            #pragma ivdep
            for (int idx = 0; idx < batch_count; idx++)
            {
                arrayA[offset + idx] = Ap2p[idx][j*lda+i];
            }
        }
    }
  
}

void memcpy_saintl2ptp(float **Ap2p, float *arrayA, int m, int batch_count) {
    int lda = m;
    for (int j = 0; j < m; j++) {
        for (int i = j; i < m; i++ ) {
            int offset = (j*(2*m-j-1)/2 + i)*batch_count;
            #pragma omp parallel for
            #pragma ivdep
            for (int idx = 0; idx < batch_count; idx++)
            {
                Ap2p[idx][j*lda+i] = arrayA[offset + idx];
            }
        }
    }
  
}


void memcpy_saptp2blkintl(float *arrayAblk, float **Ap2p, int n, int block_size, int batch_count) {
    int lda = n;
    int startpos;
    int offset;
    int numblocks = batch_count / block_size;
    int remainder = 0;
    if (batch_count % block_size != 0)
    {
        numblocks += 1;
        remainder = (batch_count % block_size);
    }
    #pragma omp parallel for
    for (int blkidx = 0; blkidx < numblocks; blkidx++) {
        int startposA = blkidx * block_size * n*(n+1)/2;
        // Remainder
        if ((blkidx == numblocks-1) && (remainder != 0)) {
            //Convert Ap2p -> A block interleave
            for (int j = 0; j < n; j++) {
                for (int i = j; i < n; i++) {
                    offset = startposA + (j*(2*n-j-1)/2 + i)*block_size;
                    #pragma vector aligned
                    for (int idx = 0; idx < remainder; idx++) {
                        arrayAblk[offset + idx] = Ap2p[blkidx * block_size + idx][j*lda+i];
                    }
                }
            }
        }  else {
            for (int j = 0; j < n; j++) {
                for (int i = j; i < n; i++) {
                    offset = startposA + (j*(2*n-j-1)/2 + i)*block_size;
                    #pragma vector aligned
                    for (int idx = 0; idx < block_size; idx++) {
                        arrayAblk[offset + idx] = Ap2p[blkidx * block_size + idx][j*lda+i];
                    }
                }
            }
        }
    }
}

void memcpy_sablkintl2ptp(float **Ap2p, float *arrayAblk, int n, int block_size, int batch_count) {
    int lda = n;
    int startpos;
    int offset;
    int numblocks = batch_count / block_size;
    int remainder = 0;
    if (batch_count % block_size != 0)
    {
        numblocks += 1;
        remainder = (batch_count % block_size);
    }
    #pragma omp parallel for
    for (int blkidx = 0; blkidx < numblocks; blkidx++) {
        int startposA = blkidx * block_size * n*(n+1)/2;
        // Remainder
        if ((blkidx == numblocks-1) && (remainder != 0)) {
            //Convert Ap2p -> A block interleave
            for (int j = 0; j < n; j++) {
                for (int i = j; i < n; i++) {
                    offset = startposA + (j*(2*n-j-1)/2 + i)*block_size;
                    #pragma vector aligned
                    for (int idx = 0; idx < remainder; idx++) {
                        Ap2p[blkidx * block_size + idx][j*lda+i] = arrayAblk[offset + idx];
                    }
                }
            }
        }  else {
            for (int j = 0; j < n; j++) {
                for (int i = j; i < n; i++) {
                    offset = startposA + (j*(2*n-j-1)/2 + i)*block_size;
                    #pragma vector aligned
                    for (int idx = 0; idx < block_size; idx++) {
                        Ap2p[blkidx * block_size + idx][j*lda+i] = arrayAblk[offset + idx];
                    }
                }
            }
        }
    }

}



void memcpy_sbintl2ptp(float **Bp2p, float *arrayB, int m, int n, int batch_count) {

    for (int pos = 0; pos < m*n; pos++)
    {
        int offset = pos*batch_count;
        #pragma omp parallel for
        #pragma ivdep
        for (int idx = 0; idx < batch_count; idx++)
        {
            Bp2p[idx][pos] = arrayB[offset + idx];
        }
    }
}


void memcpy_sbblkintl2ptp(float **Bp2p, float *arrayBblk, int m, int n, int block_size, int batch_count) {

    int blocksrequired = batch_count / block_size;
    int remainder = 0;
    int startpos;
    #pragma omp parallel for
    #pragma ivdep
    for (int blkidx = 0; blkidx < blocksrequired; blkidx++)
    {
        startpos = blkidx * block_size * m*n;
        if (blkidx == blocksrequired - 1 && remainder != 0)
        {
            // Remainders
            for (int pos = 0; pos < m*n; pos++) {
                int offset = startpos + pos*block_size;
                for (int idx = 0; idx < remainder; idx++)
                {
                    Bp2p[blkidx * block_size + idx][pos] = arrayBblk[offset + idx];
                }
            }
        }
        else
        {
            for (int pos = 0; pos < m*n; pos++)
            {
                int offset = startpos + pos*block_size;
                for (int idx = 0; idx < block_size; idx++)
                {
                    Bp2p[blkidx * block_size + idx][pos] = arrayBblk[offset + idx];
                }
            }
        }
    }  
}

void memcpy_sbptp2ptp(float **Bp2p, float **Bref, int m, int n, int batch_count) {
    for (int idx = 0; idx < batch_count; idx++)
    {
        memcpy(Bp2p[idx], Bref[idx], sizeof(float)*m*n);      
    }
}


float get_serror(float **Bref, float **Bsol, int m, int n, int batch_count) {

    float error = 0.0;
    float tmp_error;

    for (int idx = 0; idx < batch_count; idx++) {
        tmp_error = 0.0;
        for (int ij = 0; ij < m*n; ij++){
            tmp_error += fabs(Bref[idx][ij] - Bsol[idx][ij]);
        }
        if (tmp_error > error) error = tmp_error;
    }
    return error/m*n;
}
