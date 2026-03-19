#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void lm_ggml_cuda_op_set_rows(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
