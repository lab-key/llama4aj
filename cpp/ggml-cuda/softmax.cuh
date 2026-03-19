#include "common.cuh"

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void lm_ggml_cuda_op_soft_max(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_soft_max_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
