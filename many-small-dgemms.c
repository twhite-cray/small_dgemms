// many-small-dgemms.c:
// Mini-app which investigates different ways to perform many small dgemm operations.
// Looping over thousands of cublas calls is slow, so looking into other solutions.
// This functionality is very interesting to several applications, including nuccor.
// Date: Oct 15th, 2019
// Author: Justin Gage Lietz
// Contact: lietzjg@ornl.gov

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#ifdef O_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#include <omp.h>
#include <string.h> //for memcpy

// tolerance as 1.e-15 fails
#define TOL 1.e-14

#ifdef O_MAGMA
#include <cuda_runtime.h>
#include <magma_v2.h>

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
#endif

#ifdef O_HIP
#include <hip/hip_runtime_api.h>
#include "gem.h"

#define HIP_CHECK(statement) \
  { \
    const hipError_t err = (statement); \
    if (err != hipSuccess) { \
      fprintf(stderr,"Error: %s\nfailed at %s:%d: error %d: %s\n",#statement, \
          __FILE__,__LINE__,err,hipGetErrorString(err)); \
      exit(err); \
    } \
  }
#endif


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


#ifdef O_CUDA
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
    const size_t *Aoffsets, const size_t *Boffsets, const size_t *Coffsets, const size_t nBlocks, const int nIters){

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

  for(int i = 0; i <  nIters; i++){
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
  }
  double stop = omp_get_wtime();
  printf("time of cublasDgemm loop: %f\n", stop - start);
  fflush(stdout);
  cudaMemcpy(C, d_C, CBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
#endif

#ifdef O_HIP
void hip_array_dgemm(
    const double *const A, const double *const B, double *const C,
    const size_t *const Ms, const size_t *const Ns, const size_t *const Ks,
    const size_t *const Aoffsets, const size_t *const Boffsets, const size_t *const Coffsets, const size_t nBlocks, const int nIters){

  const size_t na = Aoffsets[nBlocks];
  const size_t nb = Boffsets[nBlocks];
  const size_t nc = Coffsets[nBlocks];
  const size_t matrixBytes = (na+nb+nc)*sizeof(double);
  double *a_;
  HIP_CHECK(hipMalloc((void**)&a_,matrixBytes));
  double *const b_ = a_+na;
  double *const c_ = b_+nb;
  const size_t abytes = na*sizeof(double);
  HIP_CHECK(hipHostRegister((void*)A,abytes,hipHostRegisterDefault));
  HIP_CHECK(hipMemcpyAsync(a_,A,abytes,hipMemcpyDefault,0));
  const size_t bbytes = nb*sizeof(double);
  HIP_CHECK(hipHostRegister((void*)B,bbytes,hipHostRegisterDefault));
  HIP_CHECK(hipMemcpyAsync(b_,B,nb*sizeof(double),hipMemcpyDefault,0));

  const size_t indexBytes = 3*nBlocks*sizeof(int);
  int *ms;
  HIP_CHECK(hipHostMalloc((void**)&ms,indexBytes,hipHostMallocDefault));
  int *const ns = ms+nBlocks;
  int *const ks = ns+nBlocks;

  for (int i = 0; i < nBlocks; i++) {
    ms[i] = Ms[i];
    ns[i] = Ns[i];
    ks[i] = Ks[i];
  }

  int *ms_;
  HIP_CHECK(hipMalloc((void**)&ms_,indexBytes));
  int *const ns_ = ms_+nBlocks;
  int *const ks_ = ns_+nBlocks;

  HIP_CHECK(hipMemcpyAsync(ms_,ms,indexBytes,hipMemcpyDefault,0));

  const size_t offsetBytes = 3*nBlocks*sizeof(double*);
  double **as;
  HIP_CHECK(hipHostMalloc((void**)&as,offsetBytes,hipHostMallocDefault));
  double **const bs = as+nBlocks;
  double **const cs = bs+nBlocks;
  for (int i = 0; i < nBlocks; i++) {
    as[i] = a_+Aoffsets[i];
    bs[i] = b_+Boffsets[i];
    cs[i] = c_+Coffsets[i];
  }

  double **abcs;
  HIP_CHECK(hipMalloc((void**)&abcs,offsetBytes));
  HIP_CHECK(hipMemcpy(abcs,as,offsetBytes,hipMemcpyDefault));
  const double *const *const as_ = (const double *const *)abcs;
  const double *const *const bs_ = (const double *const *)abcs+nBlocks;
  double *const *const cs_ = abcs+nBlocks+nBlocks;

  HIP_CHECK(hipDeviceSynchronize());
  const double start = omp_get_wtime();
  for (int i = 0; i < nIters; i++) hip_dgemm_vbatched(as_,bs_,cs_,ms_,ns_,ks_,nBlocks);
  HIP_CHECK(hipDeviceSynchronize());
  const double stop = omp_get_wtime();
  printf("time of hip_dgemm_vbatched: %f\n",stop-start);
  fflush(stdout);

  const size_t cbytes = nc*sizeof(double);
  HIP_CHECK(hipHostRegister(C,cbytes,hipHostRegisterDefault));
  HIP_CHECK(hipMemcpy(C,c_,nc*sizeof(double),hipMemcpyDefault));
  HIP_CHECK(hipHostUnregister(C));

  HIP_CHECK(hipFree((void*)as_));
  HIP_CHECK(hipHostFree(as));
  HIP_CHECK(hipFree(ms_));
  HIP_CHECK(hipHostFree(ms));
  HIP_CHECK(hipHostUnregister((void*)B));
  HIP_CHECK(hipHostUnregister((void*)A));
  HIP_CHECK(hipFree(a_));
}
#endif

#ifdef O_MAGMA
void magma_array_dgemm(
    const double *A, const double* B, double* C,
    const size_t *Ms, const size_t *Ns, const size_t *Ks,
    const size_t *Aoffsets, const size_t *Boffsets, const size_t *Coffsets, const size_t nBlocks, const int nIters){
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

  start = magma_sync_wtime(queue);

  for(int i = 0; i < nIters; i++){
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
  }
  stop = magma_sync_wtime(queue);

  printf("time of magmablas_dgemm_vbatched: %f\n", stop - start);
  fflush(stdout);

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
#endif


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
    const size_t *Aoffsets, const size_t *Boffsets, const size_t *Coffsets, const size_t nBlocks, const int nIters){

  for(int i = 0; i < nIters; i++){
    for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
      size_t Aoffset = Aoffsets[iBlock];
      size_t Boffset = Boffsets[iBlock];
      size_t Coffset = Coffsets[iBlock];

      host_matmul(&A[Aoffset], &B[Boffset], &C[Coffset], Ms[iBlock], Ns[iBlock], Ks[iBlock]);
    }
  }
}


void host_array_dgemm(const double *A, const double* B, double* C,
    const size_t *Ms, const size_t *Ns, const size_t *Ks,
    const size_t *Aoffsets, const size_t *Boffsets, const size_t *Coffsets, const size_t nBlocks, const int nIters){

  for(int i = 0; i < nIters; i++){
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
  for(size_t iBlock = 0; iBlock < nBlocks; iBlock++){
    size_t Aoffset = Aoffsets[iBlock];
    size_t Boffset = Boffsets[iBlock];

    if( !check_equal(&mat_A[Aoffset], &mat_B[Boffset], nrows[iBlock], ncols[iBlock]) ){
      printf("error in block: %zu\n", iBlock);
      printf("C\n");
      print_matrix( &mat_A[Aoffset], nrows[iBlock], ncols[iBlock] );
      printf("C'\n");
      print_matrix( &mat_B[Aoffset], nrows[iBlock], ncols[iBlock] );
      return iBlock;
    }
  }
  return -1;
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
#ifdef O_MAGMA
  // Start magma
  TESTING_CHECK( magma_init() );
  magma_print_environment();
#endif

  // magma seems to break at 2^16, but 2^15 blocks is fine
  size_t nBlocks = 1<<24;
  const int nIters = 1;
  size_t max_block_size = 10;
  // Small test for debugging
  //const int nIters = 1;
  //size_t nBlocks = 3;
  //size_t max_block_size = 4;

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

  printf("%d iterations with %lu matrices of size (1-%lu)x(1-%lu)\n", nIters, nBlocks, max_block_size, max_block_size);
  fflush(stdout);

  double start,stop;
  int check;
  //////////////////////////////////////////////////////////////////////
  // Tests on host
  //////////////////////////////////////////////////////////////////////

  start = omp_get_wtime();
  host_array_matmul(h_A, h_B, h_C, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks, nIters);
  stop = omp_get_wtime();
  printf("naive loop time: %f\n", stop - start);
  fflush(stdout);

  start = omp_get_wtime();
  host_array_dgemm(h_A, h_B, h_C_host, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks, nIters);
  stop = omp_get_wtime();
  printf("host blas time: %f\n", stop - start);
  fflush(stdout);

  check = check_matrix_array_equal(h_C_host, h_C, Coffsets, Coffsets, Ms_array, Ns_array, nBlocks);
  if( check >= 0 ){
    printf("check host dgemm failed: %d\n",check);
  }

  //////////////////////////////////////////////////////////////////////
  // Run tests on device
  //////////////////////////////////////////////////////////////////////

#ifdef O_CUDA
  cublasHandle_t handle;
  cublasCreate(&handle);

  start = omp_get_wtime();
  cublas_array_dgemm(handle, h_A, h_B, h_C_device, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks, nIters);
  stop = omp_get_wtime();
  printf("cublas wrapper time: %f\n", stop - start);

  check = check_matrix_array_equal(h_C_device, h_C, Coffsets, Coffsets, Ms_array, Ns_array, nBlocks);
  if( check >= 0 ){
    printf("check device dgemm failed: %d\n",check);
  }
  cublasDestroy(handle);
#endif

#ifdef O_MAGMA
  start = omp_get_wtime();
  magma_array_dgemm(h_A, h_B, h_C_magma, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks, nIters);
  stop = omp_get_wtime();
  printf("magma blas wrapper time: %f\n", stop - start);
  fflush(stdout);

  check = check_matrix_array_equal(h_C_magma, h_C, Coffsets, Coffsets, Ms_array, Ns_array, nBlocks);
  if( check >= 0 ){
    printf("magma dgemm failed: %d\n",check);
  }
  magma_finalize();
#endif

#ifdef O_HIP
  HIP_CHECK(hipDeviceSynchronize());
  start = omp_get_wtime();
  hip_init();
  HIP_CHECK(hipDeviceSynchronize());
  stop = omp_get_wtime();
  printf("hip init time: %g\n", stop - start);
  fflush(stdout);
  HIP_CHECK(hipDeviceSynchronize());
  start = omp_get_wtime();
  hip_array_dgemm(h_A, h_B, h_C_device, Ms_array, Ns_array, Ks_array, Aoffsets, Boffsets, Coffsets, nBlocks, nIters);
  HIP_CHECK(hipDeviceSynchronize());
  stop = omp_get_wtime();
  printf("hip blas wrapper time: %f\n", stop - start);
  fflush(stdout);

  check = check_matrix_array_equal(h_C_device, h_C, Coffsets, Coffsets, Ms_array, Ns_array, nBlocks);
  if ( check >= 0 ){
    printf("hip dgemm failed: %d\n",check);
    printf("A(%dx%d):\n",Ms_array[check],Ks_array[check]);
    print_matrix(h_A+Aoffsets[check],Ms_array[check],Ks_array[check]);
    printf("B(%dx%d):\n",Ks_array[check],Ns_array[check]);
    print_matrix(h_B+Boffsets[check],Ks_array[check],Ns_array[check]);
    fflush(stdout);
  }
#endif

  // Clean up
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_host);
  free(h_C_device);
  free(h_C_magma);

  return 0;
}
