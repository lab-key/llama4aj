#include "common.cuh"

#define CUDA_COUNT_EQUAL_CHUNK_SIZE 128

void lm_ggml_cuda_count_equal(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
