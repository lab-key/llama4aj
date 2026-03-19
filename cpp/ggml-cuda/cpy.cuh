#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void lm_ggml_cuda_cpy(lm_ggml_backend_cuda_context & ctx, const lm_ggml_tensor * src0, lm_ggml_tensor * src1);

void lm_ggml_cuda_dup(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
