BBLAS_BASE_DIR = /home/samuelrelton/bblas_interleaved
BBLAS_SRC_DIR = $(BBLAS_BASE_DIR)/src
BBLAS_TEST_DIR = $(BBLAS_BASE_DIR)/testing
BBLAS_INC_DIR = $(BBLAS_BASE_DIR)/include

DEPS = -I$(BBLAS_BASE_DIR)/include -I$(BBLAS_TEST_DIR)
LDFLAGS = -fopenmp
CC = gcc
CFLAGS = -c -std=c99 -DADD_ -fopenmp -O3 -ftree-vectorize -mtune=native -ffast-math -fassociative-math -fprefetch-loop-arrays
#CC = icc
#CFLAGS = -c -std=c99 -DADD_ -fopenmp -O3 -xMIC-AVX512 -ftree-vectorize -mtune=native -ffast-math -fassociative-math -fprefetch-loop-arrays
DEPS += -m64 -I${MKLROOT}/include -I$(BBLAS_INC_DIR)

# BLAS libraries
BLAS_LIB =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -lpthread -lm -ldl

# CBLAS libraries
#CBLAS_DIR       =
#CBLAS_LIB       = -L$(CBLAS_DIR)/lib
#CBLAS_INC       = -I$(CBLAS_DIR)/include

# LAPACK libraries
#LAPACK_DIR      =
#LAPACK_LIB      = -L$(LAPACK_DIR)/lib
#LAPACK_INC      = -I$(LAPACK_DIR)/include

# LAPACKE libraries
#LAPACKE_DIR     =
#LAPACKE_LIB     = -L$(LAPACKE_DIR)/lib -llapacke -llapack
#LAPACKE_INC     = -I$(LAPACKE_DIR)/include

#DEPS += $(LAPACKE_INC) $(LAPACK_INC) $(CBLAS_INC)

LDFLAGS += $(LAPACKE_LIB) $(LAPACK_LIB) $(CBLAS_LIB) $(BLAS_LIB) -lm -lgfortran


BBLAS_SRC_LIST = bblas_zgemm_batch_intl.c bblas_zgemm_batch_intl_opt.c \
                 bblas_dgemm_batch_intl.c bblas_dgemm_batch_intl_opt.c \
				 bblas_dgemm_batch_blkintl.c

BBLAS_SRC = $(addprefix $(BBLAS_SRC_DIR)/, $(BBLAS_SRC_LIST))

TEST_SRC_LIST = test_zgemm.c test_dgemm.c tune_blk_dgemm.c block_size_effect.c
TEST_SRC = $(addprefix $(BBLAS_TEST_DIR)/, $(BBLAS_TEST_LIST))

SOURCES = $(BBLAS_SRC) $(TEST_SRC)
OBJECTS = $(SOURCES:.c=.o)

all:
	make test_zgemm
	make test_dgemm
	make tune_dgemm
	make block_size_effect

.DEFAULT_GOAL := all

.c.o:
	$(CC) -c $(CFLAGS) $(DEPS) -o $@ $<

test_zgemm: $(OBJECTS)
	$(CC) $(CFLAGS) $(DEPS) $(BBLAS_TEST_DIR)/test_zgemm.c -o $(BBLAS_TEST_DIR)/test_zgemm.o
	$(CC) $(OBJECTS) $(BBLAS_TEST_DIR)/test_zgemm.o $(LDFLAGS) -o $(BBLAS_TEST_DIR)/$@

test_dgemm: $(OBJECTS)
	$(CC) $(CFLAGS) $(DEPS) $(BBLAS_TEST_DIR)/test_dgemm.c -o $(BBLAS_TEST_DIR)/test_dgemm.o
	$(CC) $(OBJECTS) $(BBLAS_TEST_DIR)/test_dgemm.o $(LDFLAGS) -o $(BBLAS_TEST_DIR)/$@

tune_dgemm: $(OBJECTS)
	$(CC) $(CFLAGS) $(DEPS) $(BBLAS_TEST_DIR)/tune_blk_dgemm.c -o $(BBLAS_TEST_DIR)/tune_blk_dgemm.o
	$(CC) $(OBJECTS) $(BBLAS_TEST_DIR)/tune_blk_dgemm.o $(LDFLAGS) -o $(BBLAS_TEST_DIR)/$@

block_size_effect: $(OBJECTS)
	$(CC) $(CFLAGS) $(DEPS) $(BBLAS_TEST_DIR)/block_size_effect.c -o $(BBLAS_TEST_DIR)/block_size_effect.o
	$(CC) $(OBJECTS) $(BBLAS_TEST_DIR)/block_size_effect.o $(LDFLAGS) -o $(BBLAS_TEST_DIR)/$@


clean:
	rm */*.o
	rm */test_zgemm
	rm */test_dgemm
	rm */tune_dgemm
	rm */block_size_effect
