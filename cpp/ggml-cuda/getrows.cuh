#include "common.cuh"

#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BACK_BLOCK_SIZE 256

void get_rows_cuda(
        const void * src0_d, lm_ggml_type src0_type, const int32_t * src1_d, void * dst_d, lm_ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream);

void lm_ggml_cuda_op_get_rows(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_get_rows_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);
