#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void hip_dgemm_vbatched(const double *const *a, const double *const *b, double *const *c,
                        const int *m, const int *n, const int *k, const int nb);
#ifdef __cplusplus
}
#endif
