#include "common.cuh"

#define CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE 256

void lm_ggml_cuda_op_conv_transpose_1d(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
