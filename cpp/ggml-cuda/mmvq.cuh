#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

void lm_ggml_cuda_mul_mat_vec_q(lm_ggml_backend_cuda_context & ctx,
    const lm_ggml_tensor * src0, const lm_ggml_tensor * src1, const lm_ggml_tensor * ids, lm_ggml_tensor * dst, const lm_ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void lm_ggml_cuda_op_mul_mat_vec_q(
    lm_ggml_backend_cuda_context & ctx,
    const lm_ggml_tensor * src0, const lm_ggml_tensor * src1, lm_ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
