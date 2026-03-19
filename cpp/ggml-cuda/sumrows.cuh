#include "common.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream);
void lm_ggml_cuda_op_sum_rows(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
