#include "common.cuh"

#define CUDA_IM2COL_BLOCK_SIZE 256

void lm_ggml_cuda_op_im2col(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
void lm_ggml_cuda_op_im2col_3d(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
