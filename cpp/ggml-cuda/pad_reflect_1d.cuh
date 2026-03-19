#include "common.cuh"

#define CUDA_PAD_REFLECT_1D_BLOCK_SIZE 256

void lm_ggml_cuda_op_pad_reflect_1d(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
