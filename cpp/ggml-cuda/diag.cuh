#include "common.cuh"

#define CUDA_DIAG_BLOCK_SIZE 256

void lm_ggml_cuda_op_diag(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
