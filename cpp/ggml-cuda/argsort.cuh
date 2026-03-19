#include "common.cuh"

void lm_ggml_cuda_op_argsort(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

#ifdef LM_GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_cub(lm_ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              lm_ggml_sort_order  order,
                              cudaStream_t     stream);
#endif  // LM_GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  lm_ggml_sort_order order,
                                  cudaStream_t    stream);
