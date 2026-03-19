#include "common.cuh"

void lm_ggml_cuda_flash_attn_ext(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

bool lm_ggml_cuda_flash_attn_ext_supported(int device, const lm_ggml_tensor * dst);
