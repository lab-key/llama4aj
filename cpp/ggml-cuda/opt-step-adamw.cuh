#include "common.cuh"

#define CUDA_OPT_STEP_ADAMW_BLOCK_SIZE 256

void lm_ggml_cuda_opt_step_adamw(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
