#include "common.cuh"

#define CUDA_POOL2D_BLOCK_SIZE 256

void lm_ggml_cuda_op_pool2d(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
