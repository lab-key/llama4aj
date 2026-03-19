#include "common.cuh"

void lm_ggml_cuda_op_norm(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_group_norm(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_rms_norm(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_rms_norm_fused(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst, lm_ggml_tensor * mul_tensor);

void lm_ggml_cuda_op_rms_norm_fused_add(lm_ggml_backend_cuda_context & ctx,
                                     lm_ggml_tensor *               dst,
                                     lm_ggml_tensor *               mul_tensor,
                                     lm_ggml_tensor *               add_tensor);

void lm_ggml_cuda_op_rms_norm_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_l2_norm(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
