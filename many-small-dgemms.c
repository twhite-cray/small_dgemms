// many-small-dgemms.c:
// Mini-app which investigates different ways to perform many small dgemm operations.
// Looping over thousands of cublas calls is slow, so looking into other solutions.
// This functionality is very interesting to several applications, including nuccor.
// Date: Oct 15th, 2019
// Author: Justin Gage Lietz
// Contact: lietzjg@ornl.gov

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <string.h> //for memcpy

#include <magma_v2.h>

// tolerance as 1.e-15 fails
#define TOL 1.e-14

// Pulled from magma test code
#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )


size_t idx(size_t i, size_t j, size_t nrows, size_t ncols){
  return j*nrows + i;
}


size_t cidx(size_t i, size_t j, size_t nrows, size_t ncols){
  return i*ncols + j;
}


void print_c_matrix(double *mat, size_t nrows, size_t ncols){
  for(int i = 0; i < nrows; i++){
    for(int j = 0; j < ncols; j++){
      printf("%f ", mat[ cidx(i,j,nrows,ncols) ]);
    }
    printf("\n");
  }
  printf("\n");
}


void print_matrix(double *mat, size_t nrows, size_t ncols){
  for(int i = 0; i < nrows; i++){
    for(int j = 0; j < ncols; j++){
      printf("%f ", mat[ idx(i,j,nrows,ncols) ]);
    }
    printf("\n");
  }
  printf("\n");
}


void print_matrix_array(double *mat_A, size_t *Aoffsets, size_t *nrows, size_t *ncols, size_t nBlocks){
  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    size_t Aoffset = Aoffsets[iBlock];
    printf("iBlock: %zu\n",iBlock);
    print_matrix( &mat_A[Aoffset], nrows[iBlock], ncols[iBlock] );
  }
}


void cblas_wrapper(double *A, double* B, double* C, size_t M_in, size_t N_in, size_t K_in){
  int M = M_in;
  int K = K_in;
  int N = N_in;
  int LDA = M;
  int LDB = K;
  int LDC = M;
  double alpha = 1.0;
  double beta = 0.0;

  cblas_dgemm(/*CBLAS*/ CblasColMajor,
      /* TRANS A */ CblasNoTrans,
      /* TRANS B */ CblasNoTrans,
      /* M */M,
      /* N */N,
      /* K */K,
      /* alpha */alpha,
      /* A */A,
      /* LDA */LDA,
      /* B  */B,
      /* LDB */LDB,
      /* BETA */beta,
      /* C */C,
      /* LDC */LDC);
}


void device_dgemm(cublasHandle_t handle, double *A, double* B, double* C, size_t M_in, size_t N_in, size_t K_in){
  int M = M_in;
  int K = K_in;
  int N = N_in;
  int LDA = M;
  int LDB = K;
  int LDC = M;
  double alpha = 1.0;
  double beta = 0.0;

  size_t ABytes = sizeof(double)*M*K;
  size_t BBytes = sizeof(double)*K*N;
  size_t CBytes = sizeof(double)*M*N;

  double *d_A;
  double *d_B;
  double *d_C;

  cudaMalloc((void**)&d_A, ABytes);
  cudaMalloc((void**)&d_B, BBytes);
  cudaMalloc((void**)&d_C, CBytes);

  cudaMemcpy(d_A, A, ABytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, BBytes, cudaMemcpyHostToDevice);

  cublasDgemm(/*cublas handle*/ handle,
      /* TRANS A */ CUBLAS_OP_N,
      /* TRANS B */ CUBLAS_OP_N,
      /* M */M,
      /* N */N,
      /* K */K,
      /* alpha */&alpha,
      /* A */d_A,
      /* LDA */LDA,
      /* B  */d_B,
      /* LDB */LDB,
      /* BETA */&beta,
      /* C */d_C,
      /* LDC */LDC);

  cudaMemcpy(C, d_C, CBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}


void cublas_array_dgemm(
    cublasHandle_t handle,
    const double *A, const double* B, double* C,
    const size_t *Ms, const size_t *Ns, const size_t *Ks,
    const size_t *Aoffsets, const size_t *Boffsets, const size_t *Coffsets, const size_t nBlocks){

  double *d_A, *d_B, *d_C;
  size_t ABytes = Aoffsets[nBlocks]*sizeof(double);
  size_t BBytes = Boffsets[nBlocks]*sizeof(double);
  size_t CBytes = Coffsets[nBlocks]*sizeof(double);

  cudaMalloc((void**)&d_A, ABytes);
  cudaMalloc((void**)&d_B, BBytes);
  cudaMalloc((void**)&d_C, CBytes);

  cudaMemcpy(d_A, A, ABytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, BBytes, cudaMemcpyHostToDevice);

  double start = omp_get_wtime();

  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    size_t Aoffset = Aoffsets[iBlock];
    size_t Boffset = Boffsets[iBlock];
    size_t Coffset = Coffsets[iBlock];

    int M = Ms[iBlock];
    int N = Ns[iBlock];
    int K = Ks[iBlock];
    int LDA = M;
    int LDB = K;
    int LDC = M;
    double alpha = 1.0;
    double beta = 0.0;
    /* char transa = 'N'; */
    /* char transb = 'N'; */

    cublasDgemm(/*cublas handle*/ handle,
        /* TRANS A */ CUBLAS_OP_N,
        /* TRANS B */ CUBLAS_OP_N,
        /* M */       M,
        /* N */       N,
        /* K */       K,
        /* alpha */   &alpha,
        /* A */       &d_A[Aoffset],
        /* LDA */     LDA,
        /* B  */      &d_B[Boffset],
        /* LDB */     LDB,
        /* BETA */    &beta,
        /* C */       &d_C[Coffset],
        /* LDC */     LDC);
  }
  double stop = omp_get_wtime();
  printf("time of cublasDgemm loop: %f\n", stop - start);
  cudaMemcpy(C, d_C, CBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}


void magma_array_dgemm(
    const double *A, const double* B, double* C,
    const size_t *Ms, const size_t *Ns, const size_t *Ks,
    const size_t *Aoffsets, const size_t *Boffsets, const size_t *Coffsets, const size_t nBlocks){
  magma_trans_t transA = MagmaNoTrans;
  magma_trans_t transB = MagmaNoTrans;
  double const* * dA_array;
  double const* * dB_array;
  double **dC_array;
  double 	alpha;
  double 	beta;
  magma_int_t* d_lddb;
  magma_int_t* d_ldda;
  magma_int_t* d_lddc;
  magma_int_t* d_m;
  magma_int_t* d_n;
  magma_int_t* d_k;
  int* h_m;
  int* h_n;
  int* h_k;

  double **hA_array;
  double **hB_array;
  double **hC_array;
  double *d_A_elems;
  double *d_B_elems;
  double *d_C_elems;

  magma_int_t batchCount;
  magma_queue_t queue;
  magma_device_t device;

  magma_getdevice( &device );
  magma_queue_create( device, &queue );

  batchCount = nBlocks;
  // Magma needs larger arrays
  TESTING_CHECK( magma_malloc((void**)&d_m, (batchCount+1)*sizeof(magma_int_t)) );
  TESTING_CHECK( magma_malloc((void**)&d_n, (batchCount+1)*sizeof(magma_int_t)) );
  TESTING_CHECK( magma_malloc((void**)&d_k, (batchCount+1)*sizeof(magma_int_t)) );

  TESTING_CHECK( magma_malloc((void**)&d_ldda, (batchCount+1)*sizeof(magma_int_t) ) );
  TESTING_CHECK( magma_malloc((void**)&d_lddb, (batchCount+1)*sizeof(magma_int_t) ) );
  TESTING_CHECK( magma_malloc((void**)&d_lddc, (batchCount+1)*sizeof(magma_int_t) ) );

  // Looks like magma_malloc works with regular c int as well.
  TESTING_CHECK( magma_malloc_cpu((void**)&h_m, (batchCount+1)*sizeof(int) ) );
  TESTING_CHECK( magma_malloc_cpu((void**)&h_n, (batchCount+1)*sizeof(int) ) );
  TESTING_CHECK( magma_malloc_cpu((void**)&h_k, (batchCount+1)*sizeof(int) ) );

  // dA_array is the array of pointers need by dgemm
  // d_A_elems are the actual mtx elements being pointed to
  // hA_array is the host side pointers that will get passed to dA_array
  TESTING_CHECK( magma_malloc( (void**)&dA_array, sizeof(double*)*batchCount ) );
  TESTING_CHECK( magma_malloc( (void**)&dB_array, sizeof(double*)*batchCount ) );
  TESTING_CHECK( magma_malloc( (void**)&dC_array, sizeof(double*)*batchCount ) );

  TESTING_CHECK( magma_malloc( (void**)&d_A_elems, sizeof(double)*Aoffsets[nBlocks] ) );
  TESTING_CHECK( magma_malloc( (void**)&d_B_elems, sizeof(double)*Boffsets[nBlocks] ) );
  TESTING_CHECK( magma_malloc( (void**)&d_C_elems, sizeof(double)*Coffsets[nBlocks] ) );

  TESTING_CHECK( magma_malloc_cpu( (void**)&hA_array, sizeof(double*)*batchCount ) );
  TESTING_CHECK( magma_malloc_cpu( (void**)&hB_array, sizeof(double*)*batchCount ) );
  TESTING_CHECK( magma_malloc_cpu( (void**)&hC_array, sizeof(double*)*batchCount ) );

  // magma needs 32-bit int, while main was using 64
  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    h_m[iBlock] = Ms[iBlock];
    h_n[iBlock] = Ns[iBlock];
    h_k[iBlock] = Ks[iBlock];
  }

  magma_setvector(batchCount, sizeof(magma_int_t), h_m, 1, d_m, 1, queue);
  magma_setvector(batchCount, sizeof(magma_int_t), h_n, 1, d_n, 1, queue);
  magma_setvector(batchCount, sizeof(magma_int_t), h_k, 1, d_k, 1, queue);
  magma_setvector(batchCount, sizeof(magma_int_t), h_m, 1, d_ldda, 1, queue);
  magma_setvector(batchCount, sizeof(magma_int_t), h_k, 1, d_lddb, 1, queue);
  magma_setvector(batchCount, sizeof(magma_int_t), h_m, 1, d_lddc, 1, queue);

  magma_setvector(Aoffsets[nBlocks], sizeof(double), A, 1, d_A_elems, 1, queue);
  magma_setvector(Boffsets[nBlocks], sizeof(double), B, 1, d_B_elems, 1, queue);

  size_t Aoffset;
  size_t Boffset;
  size_t Coffset;

  for(int iBlock = 0; iBlock < nBlocks; iBlock++){
    Aoffset = Aoffsets[iBlock];
    Boffset = Boffsets[iBlock];
    Coffset = Coffsets[iBlock];

    hA_array[iBlock] = d_A_elems + Aoffset;
    hB_array[iBlock] = d_B_elems + Boffset;
    hC_array[iBlock] = d_C_elems + Coffset;
  }

  magma_setvector(batchCount, sizeof(double*), hA_array, 1, dA_array, 1, queue);
  magma_setvector(batchCount, sizeof(double*), hB_array, 1, dB_array, 1, queue);
  magma_setvector(batchCount, sizeof(double*), hC_array, 1, dC_array, 1, queue);

  alpha = 1.0;
  beta = 0.0;
  batchCount = nBlocks;

  double start,stop;

  start = omp_get_wtime();

  magmablas_dgemm_vbatched(	      transA,
      /* magma_trans_t */ 	      transB,
      /* magma_int_t * */         d_m,
      /* magma_int_t * */	        d_n,
      /* magma_int_t * */	        d_k,
      /* double */	              alpha,
      /* double const *const * */	dA_array,
      /* magma_int_t * */	        d_ldda,
      /* double const *const * */	dB_array,
      /* magma_int_t * */	        d_lddb,
      /* double */	              beta,
      /* double ** */	            dC_array,
      /* magma_int_t * */	        d_lddc,
      /* magma_int_t */	          batchCount,
      /* magma_queue_t */	        queue);
  stop = omp_get_wtime();

  printf("time of magmablas_dgemm_vbatched: %f\n", stop - start);

  magma_getvector(Coffsets[nBlocks], sizeof(double), d_C_elems, 1, C, 1, queue);

  // Clean up
  magma_queue_destroy( queue );

  TESTING_CHECK( magma_free(d_m) );
  TESTING_CHECK( magma_free(d_n) );
  TESTING_CHECK( magma_free(d_k) );

  TESTING_CHECK( magma_free(d_ldda) );
  TESTING_CHECK( magma_free(d_lddb) );
  TESTING_CHECK( magma_free(d_lddc) );

  TESTING_CHECK( magma_free_cpu(h_m) );
  TESTING_CHECK( magma_free_cpu(h_n) );
  TESTING_CHECK( magma_free_cpu(h_k) );

  TESTING_CHECK( magma_free(dA_array) );
  TESTING_CHECK( magma_free(dB_array) );
  TESTING_CHECK( magma_free(dC_array) );

  TESTING_CHECK( magma_free(d_A_elems) );
  TESTING_CHECK( magma_free(d_B_elems) );
  TESTING_CHECK( magma_free(d_C_elems) );

  TESTING_CHECK( magma_free_cpu(hA_array) );
  TESTING_CHECK( magma_free_cpu(hB_array) );
  TESTING_CHECK( magma_free_cpu(hC_array) );
}


void host_matmul(const double *A, const double* B, double* C, const size_t M, const size_t N, const size_t K){
  for(size_t i = 0; i < M; i++){
    for(size_t j = 0; j < N; j++){
      C[ idx(i,j,M,N) ] = 0.0;
      for(size_t k = 0; k < K; k++){
        C[ idx(i,j,M,N) ] += A[ idx(i,k,M,K) ] * B[ idx(k,j,K,N) ];
      }
    }
  }
}


void host_array_matmul(const double *A, const double* B, double* C,
    const size_t *Ms, const size_t *Ns, const size_t *Ks,
    const size_t *Aoffsets, const size_t *Boffsets, const size_t *Coffsets, const size_t nBlocks){

  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    size_t Aoffset = Aoffsets[iBlock];
    size_t Boffset = Boffsets[iBlock];
    size_t Coffset = Coffsets[iBlock];

    host_matmul(&A[Aoffset], &B[Boffset], &C[Coffset], Ms[iBlock], Ns[iBlock], Ks[iBlock]);
  }
}


void host_array_dgemm(const double *A, const double* B, double* C,
    const size_t *Ms, const size_t *Ns, const size_t *Ks,
    const size_t *Aoffsets, const size_t *Boffsets, const size_t *Coffsets, const size_t nBlocks){

  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    size_t Aoffset = Aoffsets[iBlock];
    size_t Boffset = Boffsets[iBlock];
    size_t Coffset = Coffsets[iBlock];

    int M = Ms[iBlock];
    int N = Ns[iBlock];
    int K = Ks[iBlock];
    int LDA = M;
    int LDB = K;
    int LDC = M;
    double alpha = 1.0;
    double beta = 0.0;
    /* char transa = 'N'; */
    /* char transb = 'N'; */

    cblas_dgemm(/*CBLAS*/ CblasColMajor,
        /* TRANS A */ CblasNoTrans,
        /* TRANS B */ CblasNoTrans,
        /* M */       M,
        /* N */       N,
        /* K */       K,
        /* alpha */   alpha,
        /* A */       &A[Aoffset],
        /* LDA */     LDA,
        /* B  */      &B[Boffset],
        /* LDB */     LDB,
        /* BETA */    beta,
        /* C */       &C[Coffset],
        /* LDC */     LDC);
  }
}


void host_transpose(double *mat_in, double *mat_out, size_t nrows_in, size_t ncols_in){
  for(size_t i = 0; i < nrows_in; i++){
    for(size_t j = 0; j < ncols_in; j++){
      mat_out[ idx(j,i,ncols_in,nrows_in) ] = mat_in[ idx(i,j,nrows_in, ncols_in) ];
    }
  }
}


int check_equal(double *mat_A, double *mat_B, size_t nrows, size_t ncols){
  int flag = 1;
  for(size_t i = 0; i < nrows; i++){
    for(size_t j = 0; j < ncols; j++){
      if( fabs( mat_A[ idx(i,j,nrows,ncols) ] - mat_B[ idx(i,j,nrows,ncols) ]) > TOL ){
        flag = 0;
        return flag;
      }
    }
  }
  return flag;
}


int check_matrix_array_equal(double *mat_A, double *mat_B, size_t *Aoffsets, size_t *Boffsets, size_t *nrows, size_t *ncols, size_t nBlocks){
  int flag = 1;
  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    size_t Aoffset = Aoffsets[iBlock];
    size_t Boffset = Boffsets[iBlock];

    flag *= check_equal( &mat_A[Aoffset], &mat_B[Boffset], nrows[iBlock], ncols[iBlock] );
    if( !check_equal(&mat_A[Aoffset], &mat_B[Boffset], nrows[iBlock], ncols[iBlock]) ){
      printf("error in block: %zu\n", iBlock);
      printf("A\n");
      print_matrix( &mat_A[Aoffset], nrows[iBlock], ncols[iBlock] );
      printf("B\n");
      print_matrix( &mat_B[Aoffset], nrows[iBlock], ncols[iBlock] );
      return flag;
    }
  }
  return flag;
}


void generate_random_array(size_t* array, size_t nBlocks, size_t max_size){
  size_t rando;

  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    rando = rand()%max_size + 1;
    array[iBlock] = rando;
  }
}


void load_random_matrix(double* mat_in, size_t nrows, size_t ncols){
  double rando;

  for(size_t i = 0; i < nrows; i++){
    for(size_t j = 0; j < ncols; j++){
      rando= (double) rand()/RAND_MAX*2.0 - 1.0;
      mat_in[ idx(i,j,nrows,ncols) ] = rando;
    }
  }
}


int main(){
  // Start magma
  TESTING_CHECK( magma_init() );
  magma_print_environment();

  // magma seems to break at 2^16, but 2^15 blocks is fine
  size_t nBlocks = 1<<25;
  size_t max_block_size = 10;
  // Small test for debugging
  // size_t nBlocks = 3;
  // size_t max_block_size = 4;

  size_t *Ms_array;
  size_t *Ns_array;
  size_t *Ks_array;
  size_t *Aoffsets;
  size_t *Boffsets;
  size_t *Coffsets;

  Ms_array = (size_t*)malloc(sizeof(size_t)*nBlocks);
  Ns_array = (size_t*)malloc(sizeof(size_t)*nBlocks);
  Ks_array = (size_t*)malloc(sizeof(size_t)*nBlocks);

  Aoffsets = (size_t*)malloc(sizeof(size_t)*(nBlocks+1));
  Boffsets = (size_t*)malloc(sizeof(size_t)*(nBlocks+1));
  Coffsets = (size_t*)malloc(sizeof(size_t)*(nBlocks+1));

  /* srand( time(NULL) ); */
  // Reproducible seed
  srand( 100 );

  generate_random_array(Ms_array,nBlocks,max_block_size);
  generate_random_array(Ns_array,nBlocks,max_block_size);
  generate_random_array(Ks_array,nBlocks,max_block_size);

  Aoffsets[0] = 0;
  Boffsets[0] = 0;
  Coffsets[0] = 0;

  // Offset array elements indicate starting location of each block
  // Final array element indicates where the next block WOULD start
  // This is needed to count the total number of elements
  for(size_t iBlock = 1; iBlock < nBlocks + 1; iBlock++){
    Aoffsets[iBlock] = Aoffsets[iBlock-1] + Ms_array[iBlock-1] * Ks_array[iBlock-1];
    Boffsets[iBlock] = Boffsets[iBlock-1] + Ks_array[iBlock-1] * Ns_array[iBlock-1];
    Coffsets[iBlock] = Coffsets[iBlock-1] + Ms_array[iBlock-1] * Ns_array[iBlock-1];
  }

  size_t ABytes = Aoffsets[nBlocks]*sizeof(double);
  size_t BBytes = Boffsets[nBlocks]*sizeof(double);
  size_t CBytes = Coffsets[nBlocks]*sizeof(double);

  double *h_A;
  double *h_B;
  double *h_C;
  double *h_C_host;
  double *h_C_device;
  double *h_C_magma;

  h_A = (double*)malloc(ABytes);
  h_B = (double*)malloc(BBytes);
  h_C = (double*)malloc(CBytes);
  h_C_host = (double*)malloc(CBytes);
  h_C_device = (double*)malloc(CBytes);
  h_C_magma = (double*)malloc(CBytes);

  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    size_t Aoffset = Aoffsets[iBlock];
    size_t Boffset = Boffsets[iBlock];

    load_random_matrix(&h_A[Aoffset], Ms_array[iBlock], Ks_array[iBlock]);
    load_random_matrix(&h_B[Boffset], Ks_array[iBlock], Ns_array[iBlock]);
  }

  double start,stop;
  int check;
  //////////////////////////////////////////////////////////////////////
  // Tests on host
  //////////////////////////////////////////////////////////////////////

  start = omp_get_wtime();
  host_array_matmul(h_A, h_B, h_C, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks);
  stop = omp_get_wtime();
  printf("naive loop time: %f\n", stop - start);

  start = omp_get_wtime();
  host_array_dgemm(h_A, h_B, h_C_host, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks);
  stop = omp_get_wtime();
  printf("host blas time: %f\n", stop - start);

  check = check_matrix_array_equal(h_C_host, h_C, Coffsets, Coffsets, Ms_array, Ns_array, nBlocks);
  if( check == 0 ){
    printf("check host dgemm failed: %d\n",check);
  }

  //////////////////////////////////////////////////////////////////////
  // Run tests on device
  //////////////////////////////////////////////////////////////////////

  cublasHandle_t handle;
  cublasCreate(&handle);

  start = omp_get_wtime();
  cublas_array_dgemm(handle, h_A, h_B, h_C_device, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks);
  stop = omp_get_wtime();
  printf("cublas wrapper time: %f\n", stop - start);

  check = check_matrix_array_equal(h_C_device, h_C, Coffsets, Coffsets, Ms_array, Ns_array, nBlocks);
  if( check == 0 ){
    printf("check device dgemm failed: %d\n",check);
  }

  start = omp_get_wtime();
  magma_array_dgemm(h_A, h_B, h_C_magma, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks);
  stop = omp_get_wtime();
  printf("magma blas wrapper time: %f\n", stop - start);

  check = check_matrix_array_equal(h_C_magma, h_C, Coffsets, Coffsets, Ms_array, Ns_array, nBlocks);
  if( check == 0 ){
    printf("magma dgemm failed: %d\n",check);
  }

  // Clean up
  cublasDestroy(handle);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_host);
  free(h_C_device);
  free(h_C_magma);
  magma_finalize();

  return 0;
}
