#include "common.cuh"

#define CUDA_SOFTCAP_BLOCK_SIZE 256

void lm_ggml_cuda_op_softcap(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst, lm_ggml_tensor * src);
