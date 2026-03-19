#include "common.cuh"

#define CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE 256

void lm_ggml_cuda_op_timestep_embedding(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
