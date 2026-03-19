#include "common.cuh"

void lm_ggml_cuda_op_repeat(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
void lm_ggml_cuda_op_add(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
void lm_ggml_cuda_op_sub(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
void lm_ggml_cuda_op_mul(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
void lm_ggml_cuda_op_div(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_repeat_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_fused_add(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst, int n_fuse);
