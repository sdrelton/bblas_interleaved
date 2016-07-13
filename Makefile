BBLAS_BASE_DIR = /home/srelton/NLAFET/bblas_interleaved
BBLAS_SRC_DIR = $(BBLAS_BASE_DIR)/src
BBLAS_TEST_DIR = $(BBLAS_BASE_DIR)/testing

DEPS = -O3 -I$(BBLAS_BASE_DIR)/include -I$(BBLAS_TEST_DIR)
LDFLAGS = -fopenmp
CC = gcc
CFLAGS = -c -std=c99 -DADD_ -Wall -pedantic -fopenmp -g
DEPS += -m64 -I${MKLROOT}/include

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

DEPS += $(LAPACKE_INC) $(LAPACK_INC) $(CBLAS_INC)

LDFLAGS += $(LAPACKE_LIB) $(LAPACK_LIB) $(CBLAS_LIB) $(BLAS_LIB) -lm -lgfortran






BBLAS_SRC_LIST = bblas_zgemm_batch_intl.c bblas_zgemm_batch_intl_opt.c

BBLAS_SRC = $(addprefix $(BBLAS_SRC_DIR)/, $(BBLAS_SRC_LIST))

TEST_SRC_LIST = test_zgemm.c
TEST_SRC = $(addprefix $(BBLAS_TEST_DIR)/, $(BBLAS_TEST_LIST))

SOURCES = $(BBLAS_SRC) $(TEST_SRC)
SOURCES_Z = $(SOURCES)
OBJECTS_Z = $(SOURCES_Z:.c=.o)

all:
	@echo "$(SOURCES)"
	@echo "$(OBJECTS_Z)"
	make test_zgemm

.DEFAULT_GOAL := all

test_zgemm: $(OBJECTS_Z)
	$(CC) $(CFLAGS) $(DEPS) $(BBLAS_TEST_DIR)/test_zgemm.c -o $(BBLAS_TEST_DIR)/test_zgemm.o
	$(CC) $(OBJECTS_Z) $(BBLAS_TEST_DIR)/test_zgemm.o $(LDFLAGS) -o $(BBLAS_TEST_DIR)/$@

.c.o:
	$(CC) $(CFLAGS) $(DEPS) $<   -o $@

clean:
	rm */*.o
	rm */testzgemm
