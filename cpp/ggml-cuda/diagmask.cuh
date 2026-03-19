#include "common.cuh"

#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32

void lm_ggml_cuda_op_diag_mask_inf(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
