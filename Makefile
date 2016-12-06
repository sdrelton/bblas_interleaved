include make.inc.icc

BBLAS_SRC_LIST = bblas_dgemm_intl.c bblas_dgemm_intl_opt.c \
	 	 bblas_dgemm_blkintl_expert.c bblas_dgemm_blkintl.c bblas_dtrsm_intl.c \
         bblas_dtrsm_blkintl.c bblas_dtrsm_blkintl_expert.c \
		 bblas_dtrsm_intl_expert.c bblas_dpotrf_blkintl.c \
		 bblas_dpotrf_blkintl_expert.c bblas_dpotrf_intl.c \
		 bblas_dpotrf_intl_expert.c \
		 bblas_dposv_intl_expert.c bblas_dposv_intl.c \
		 bblas_dposv_blkintl_expert.c bblas_dposv_blkintl.c \


BBLAS_SRC = $(addprefix $(BBLAS_SRC_DIR)/, $(BBLAS_SRC_LIST))
COMMON_SRC = $(BBLAS_SRC) $(BBLAS_TEST_DIR)/test_dconversion.c 


#Create object files test_dtrsm binary 
SOURCES_DTRSM = $(COMMON_SRC) $(BBLAS_TEST_DIR)/test_dtrsm.c 
OBJECTS_DTRSM = $(SOURCES_DTRSM:.c=.o)

# Single precision
SOURCES_STRSM = $(subst _d,_s,$(SOURCES_DTRSM))
OBJECTS_STRSM = $(SOURCES_STRSM:.c=.o)

#Create object files test_dpotrf binary 
SOURCES_DPOTRF = $(COMMON_SRC) $(BBLAS_TEST_DIR)/test_dpotrf.c 
OBJECTS_DPOTRF = $(SOURCES_DPOTRF:.c=.o)

# Single precision
SOURCES_SPOTRF = $(subst _d,_s,$(SOURCES_DPOTRF))
OBJECTS_SPOTRF = $(SOURCES_SPOTRF:.c=.o)


#Create object files test_dpotrf binary 
SOURCES_DPOSV = $(COMMON_SRC) $(BBLAS_TEST_DIR)/test_dposv.c 
OBJECTS_DPOSV = $(SOURCES_DPOSV:.c=.o)

# Single precision
SOURCES_SPOSV = $(subst _d,_s,$(SOURCES_DPOSV))
OBJECTS_SPOSV = $(SOURCES_SPOSV:.c=.o)

#Create object files for test_dgemm binary
SOURCES_DGEMM = $(COMMON_SRC) $(BBLAS_TEST_DIR)/test_dgemm.c 
OBJECTS_DGEMM = $(SOURCES_DGEMM:.c=.o)

# Single precision
SOURCES_SGEMM = $(subst _d,_s,$(SOURCES_DGEMM))
OBJECTS_SGEMM = $(SOURCES_SGEMM:.c=.o)

#Create object files for tune_dgemm binary
SOURCES_TUNE = $(COMMON_SRC) $(BBLAS_TEST_DIR)/tune_blk_dgemm.c
OBJECTS_TUNE = $(SOURCES_TUNE:.c=.o)

#Create object files for  binary block_size_effect
SOURCES_BLK = $(COMMON_SRC) $(BBLAS_TEST_DIR)/block_size_effect.c
OBJECTS_BLK = $(SOURCES_BLK:.c=.o)

all:
	make test_dgemm test_sgemm  test_dtrsm test_strsm test_dpotrf test_spotrf test_dposv test_sposv 

.DEFAULT_GOAL := all

test_dgemm: $(OBJECTS_DGEMM)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_DGEMM) $(LDFLAGS) -o $@

test_sgemm: $(OBJECTS_SGEMM)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_SGEMM) $(LDFLAGS) -o $@

tune_dgemm: $(OBJECTS_TUNE)
	 cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_TUNE) $(LDFLAGS) -o $@

block_size_effect: $(OBJECTS_BLK)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_BLK) $(LDFLAGS) -o $@

test_dtrsm: $(OBJECTS_DTRSM)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_DTRSM) $(LDFLAGS) -o $@

test_strsm: $(OBJECTS_STRSM)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_STRSM) $(LDFLAGS) -o $@

test_dpotrf: $(OBJECTS_DPOTRF)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_DPOTRF) $(LDFLAGS) -o $@

test_spotrf: $(OBJECTS_SPOTRF)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_SPOTRF) $(LDFLAGS) -o $@

test_dposv: $(OBJECTS_DPOSV)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_DPOSV) $(LDFLAGS) -o $@

test_sposv: $(OBJECTS_SPOSV)
	cd $(BBLAS_TEST_DIR); $(CC) $(OBJECTS_SPOSV) $(LDFLAGS) -o $@

.c.o:
	$(CC) -c $(CFLAGS) $(DEPS) $< -o $@ 

.SILENT: clean
clean:
	-@rm $(BBLAS_SRC_DIR)/*.o $(BBLAS_SRC_DIR)/*~ 2>/dev/null || true
	cd  $(BBLAS_TEST_DIR); rm test_dgemm test_sgemm tune_dgemm block_size_effect test_dtrsm test_strsm test_dpotrf test_dposv test_spotrf test_sposv *~ *.o 2>/dev/null || true
