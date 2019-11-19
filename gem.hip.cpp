#include "gem.h"

#include <hip/hip_runtime.h>


union double_int2 {
    double d;
      int2 i;
};


template <int lane>
__device__ double swizzle(const double_int2 x)
{
    constexpr int pattern = 0x18 | (lane<<5);
      return double_int2{.i = {__hip_ds_swizzle(x.i.x,pattern),__hip_ds_swizzle(x.i.y,pattern)}}.d;
}


template <int lane>
__device__ double swizzle8(const double_int2 x)
{
    constexpr int pattern = 7 | (lane<<8);
      return double_int2{.i = {__hip_ds_swizzle(x.i.x,pattern),__hip_ds_swizzle(x.i.y,pattern)}}.d;
}


__launch_bounds__(64)
__global__ void amethyst(const double *const *const ag, const double *const *const bg, double *const *const cg,
                       const int *const mg, const int *const ng, const int *const kg)
{
  const double *const a = ag[blockIdx.x];
  const double *const b = bg[blockIdx.x];
  double *const c = cg[blockIdx.x];

  const int m = mg[blockIdx.x];
  const int n = ng[blockIdx.x];
  const int k = kg[blockIdx.x];

  for (int q = 0; q < n; q += 8) {
    const int j = q+threadIdx.y;
    for (int p = 0; p < m; p += 8) {
      const int i = p+threadIdx.x;
      double cr = 0;
      for (int r = 0; r < k; r += 8) {

        const int ry = r+threadIdx.y;
        const double ar = ((i < m) && (ry < k)) ? a[i+ry*m] : 0;
        const int rx = r+threadIdx.x;
        const double_int2 br{.d = ((j < n) && (rx < k)) ? b[rx+j*k] : 0};
        const double_int2 alo{.d = __shfl_up(ar,32)};
        const double_int2 ahi{.d = __shfl_down(ar,32)};

        cr += swizzle8<0>(alo)*swizzle<0>(br);
        cr += swizzle8<1>(alo)*swizzle<1>(br);
        cr += swizzle8<2>(alo)*swizzle<2>(br);
        cr += swizzle8<3>(alo)*swizzle<3>(br);
        cr += swizzle8<0>(ahi)*swizzle<4>(br);
        cr += swizzle8<1>(ahi)*swizzle<5>(br);
        cr += swizzle8<2>(ahi)*swizzle<6>(br);
        cr += swizzle8<3>(ahi)*swizzle<7>(br);
      }
      if ((i < m) && (j < n)) c[i+j*m] = cr;
    }
  }
}


__global__ void noop() { }


extern "C" {
  void hip_dgemm_vbatched(const double *const *const a, const double *const *const b, double *const *const c,
                          const int *const m, const int *const n, const int *const k, const int nb)
  {
    hipLaunchKernelGGL(amethyst,nb,dim3(8,8),0,0,a,b,c,m,n,k);
  }
  void hip_init()
  {
    hipLaunchKernelGGL(noop,1,1,0,0);
  }
}

