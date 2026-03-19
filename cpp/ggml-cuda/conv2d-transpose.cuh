#include "common.cuh"

#define CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE 256
void lm_ggml_cuda_conv_2d_transpose_p0(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
