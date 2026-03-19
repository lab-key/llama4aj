#include "common.cuh"

#define CUDA_WKV_BLOCK_SIZE 64

void lm_ggml_cuda_op_rwkv_wkv6(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_rwkv_wkv7(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
