#include "gem.h"

#include <hip/hip_runtime.h>


__launch_bounds__(64)
__global__ void amethyst(const double *const *const ag, const double *const *const bg, double *const *const cg,
                       const int *const mg, const int *const ng, const int *const kg)
{
  const double *const a = ag[blockIdx.x];
  const double *const b = bg[blockIdx.x];
  double *const c = cg[blockIdx.x];

  __shared__ double as[8][8];
  __shared__ double bs[8][8];
  
  const int m = mg[blockIdx.x];
  const int n = ng[blockIdx.x];
  const int k = kg[blockIdx.x];

  for (int q = 0; q < n; q += 8) {
    const int j = q+threadIdx.y;
    for (int p = 0; p < m; p += 8) {
      const int i = p+threadIdx.x;
      double cr = 0;
      for (int r = 0; r < k; r += 8) {
        __syncthreads();
        const int ry = r+threadIdx.y;
        as[threadIdx.y][threadIdx.x] = ((i < m) && (ry < k)) ? a[i+ry*m] : 0;
        const int rx = r+threadIdx.x;
        bs[threadIdx.y][threadIdx.x] = ((j < n) && (rx < k)) ? b[rx+j*k] : 0;
        __syncthreads();
        for (int l = 0; l < 8; l++) {
          cr += as[l][threadIdx.x]*bs[threadIdx.y][l];
        }
      }
      if ((i < m) && (j < n)) c[i+j*m] = cr;
    }
  }
}


extern "C" {
  void hip_dgemm_vbatched(const double *const *const a, const double *const *const b, double *const *const c,
                          const int *const m, const int *const n, const int *const k, const int nb)
  {
    hipLaunchKernelGGL(amethyst,nb,dim3(8,8),0,0,a,b,c,m,n,k);
  }
}
