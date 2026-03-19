#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void lm_ggml_cuda_op_rope(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_rope_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_rope_fused(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst, lm_ggml_tensor * set_rows);
