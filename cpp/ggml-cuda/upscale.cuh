#include "common.cuh"

#define CUDA_UPSCALE_BLOCK_SIZE 256

void lm_ggml_cuda_op_upscale(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
