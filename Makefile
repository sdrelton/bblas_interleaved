include make.inc.icc

BBLAS_SRC_LIST = bblas_zgemm_batch_intl.c bblas_zgemm_batch_intl_opt.c \
                 bblas_dgemm_batch_intl.c bblas_dgemm_batch_intl_opt.c \
	 	 bblas_dgemm_batch_blkintl.c bblas_dtrsm_batch_intl.c \
                 bblas_dtrsm_batch_blkintl.c bblas_dtrsm_batch_blkintl_expert.c \
		 bblas_dtrsm_batch_intl_expert.c	


BBLAS_SRC = $(addprefix $(BBLAS_SRC_DIR)/, $(BBLAS_SRC_LIST))
COMMON_SRC = $(BBLAS_SRC) $(BBLAS_TEST_DIR)/test_dconversion.c 

#Create object files test_dtrsm binary 
SOURCES_DTRSM = $(COMMON_SRC) $(BBLAS_TEST_DIR)/test_dtrsm.c 
OBJECTS_DTRSM = $(SOURCES_DTRSM:.c=.o)

#Create object files for test_dgemm binary
SOURCES_DGEMM = $(COMMON_SRC) $(BBLAS_TEST_DIR)/test_dgemm.c 
OBJECTS_DGEMM = $(SOURCES_DGEMM:.c=.o)

#Create object files for tune_dgemm binary
SOURCES_TUNE = $(COMMON_SRC) $(BBLAS_TEST_DIR)/tune_blk_dgemm.c
OBJECTS_TUNE = $(SOURCES_TUNE:.c=.o)

#Create object files for  binary block_size_effect
SOURCES_BLK = $(COMMON_SRC) $(BBLAS_TEST_DIR)/block_size_effect.c
OBJECTS_BLK = $(SOURCES_BLK:.c=.o)

all:
	make test_dgemm tune_dgemm block_size_effect test_dtrsm

.DEFAULT_GOAL := all

test_dgemm: $(OBJECTS_DGEMM)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_DGEMM) $(LDFLAGS) -o $@

tune_dgemm: $(OBJECTS_TUNE)
	 cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_TUNE) $(LDFLAGS) -o $@

block_size_effect: $(OBJECTS_BLK)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_BLK) $(LDFLAGS) -o $@

test_dtrsm: $(OBJECTS_DTRSM)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_DTRSM) $(LDFLAGS) -o $@

.c.o:
	$(CC) -c $(CFLAGS) $(DEPS) $< -o $@ 

.SILENT: clean
clean:
	-@rm $(BBLAS_SRC_DIR)/*.o $(BBLAS_SRC_DIR)/*~ 2>/dev/null || true
	cd  $(BBLAS_TEST_DIR); @rm test_zgemm test_dgemm tune_dgemm block_size_effect test_dtrsm *~ 2>/dev/null || true
