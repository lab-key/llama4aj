#include "common.cuh"

#define CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE 256

void lm_ggml_cuda_cross_entropy_loss(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_cross_entropy_loss_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
