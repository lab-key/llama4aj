#include "common.cuh"

void sum_f32_cuda(lm_ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream);

void lm_ggml_cuda_op_sum(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
