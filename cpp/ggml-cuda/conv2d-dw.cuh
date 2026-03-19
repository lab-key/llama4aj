#pragma once
#include "common.cuh"

#define CUDA_CONV2D_DW_BLOCK_SIZE 256
void lm_ggml_cuda_op_conv2d_dw(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
