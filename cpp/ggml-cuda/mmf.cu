#include "ggml.h"
#include "mmf.cuh"
#include "mmid.cuh"


void lm_ggml_cuda_mul_mat_f(lm_ggml_backend_cuda_context & ctx, const lm_ggml_tensor * src0, const lm_ggml_tensor * src1, const lm_ggml_tensor * ids, lm_ggml_tensor * dst) {
    LM_GGML_ASSERT(        src1->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(!ids ||  ids->type == LM_GGML_TYPE_I32);
    LM_GGML_ASSERT(         dst->type == LM_GGML_TYPE_F32);


    LM_GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = lm_ggml_type_size(src0->type);
    const size_t ts_src1 = lm_ggml_type_size(src1->type);
    const size_t ts_dst  = lm_ggml_type_size(dst->type);

    LM_GGML_ASSERT(ne13 == ne3);

    LM_GGML_ASSERT(        nb00       == ts_src0);
    LM_GGML_ASSERT(        nb10       == ts_src1);
    LM_GGML_ASSERT(!ids || ids->nb[0] == lm_ggml_type_size(ids->type));
    LM_GGML_ASSERT(        nb0        == ts_dst);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t ids_s0 = ids ? ids->nb[0] / lm_ggml_type_size(ids->type) : 0;
    const int64_t ids_s1 = ids ? ids->nb[1] / lm_ggml_type_size(ids->type) : 0;

    mmf_ids_data ids_info{};
    mmf_ids_data * ids_info_ptr = nullptr;
    lm_ggml_cuda_pool_alloc<int32_t> ids_src_compact_dev;
    lm_ggml_cuda_pool_alloc<int32_t> ids_dst_compact_dev;
    lm_ggml_cuda_pool_alloc<int32_t> expert_bounds_dev;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_dst      = ids ? ne1 : ne2;

    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1 : s2;

    int64_t stride_channel_y         = ids ? s11  : s12;
    int64_t nchannels_y              = ids ? ne11 : ne12;

    //mul_mat_id: handle broadcast
    if (ids && nchannels_y == 1) {
        stride_channel_y = 0;
        nchannels_y      = ids->ne[0];
    }

    if (ids && ncols_dst > 16) {
        const int64_t n_expert_used = ids->ne[0];
        const int64_t n_experts     = ne02;
        const int64_t n_tokens      = ne12;
        const int64_t ne_get_rows   = n_tokens * n_expert_used;

        ids_src_compact_dev.alloc(ctx.pool(), ne_get_rows);
        ids_dst_compact_dev.alloc(ctx.pool(), ne_get_rows);
        expert_bounds_dev.alloc(ctx.pool(), n_experts + 1);

        const int si1  = static_cast<int>(ids_s1);
        const int sis1 = static_cast<int>(src1->nb[2] / src1->nb[1]);

        LM_GGML_ASSERT(sis1 > 0);

        lm_ggml_cuda_launch_mm_ids_helper(ids_d, ids_src_compact_dev.get(), ids_dst_compact_dev.get(), expert_bounds_dev.get(),
            static_cast<int>(n_experts), static_cast<int>(n_tokens), static_cast<int>(n_expert_used), static_cast<int>(ne11), si1, sis1, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        ids_info.ids_src_compact   = ids_src_compact_dev.get();
        ids_info.ids_dst_compact   = ids_dst_compact_dev.get();
        ids_info.expert_bounds_dev = expert_bounds_dev.get();
        ids_info.n_experts         = static_cast<int>(n_experts);
        ids_info.sis1              = sis1;
        ids_info_ptr = &ids_info;
    }

    switch (src0->type) {
        case LM_GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            constexpr int vals_per_T = 1;
            mul_mat_f_switch_cols_per_block(
                src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        case LM_GGML_TYPE_F16: {
            const half2 * src0_d = (const half2 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_cols_per_block(
                src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        case LM_GGML_TYPE_BF16: {
            const nv_bfloat162 * src0_d = (const nv_bfloat162 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_cols_per_block(
                src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        default:
            LM_GGML_ABORT("unsupported type: %s", lm_ggml_type_name(src0->type));
    }
}

bool lm_ggml_cuda_should_use_mmf(enum lm_ggml_type type, int cc, int warp_size, const int64_t * src0_ne,
        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {
    if (lm_ggml_is_quantized(type)) {
        return false;
    }

    const size_t ts = lm_ggml_type_size(type);
    if (src0_ne[0] % (warp_size * (4/ts)) != 0) {
        return false;
    }

    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < LM_GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }
    if (src0_ne[1] % MMF_ROWS_PER_BLOCK != 0) {
        return false;
    }

    if (mul_mat_id) {
        if (src0_ne[1] <= 1024 && src1_ncols > 512) {
            return false;
        } else if(src0_ne[1] > 1024 && src1_ncols > 128) {
            return false;
        }
    } else {
        if (LM_GGML_CUDA_CC_IS_RDNA3_0(cc) && src1_ncols > 8) {
            return false;
        } else if (src1_ncols > 16) {
            return false;
        }
    }

    switch (type) {
        case LM_GGML_TYPE_F32:
            return ampere_mma_available(cc);
        case LM_GGML_TYPE_F16:
            return volta_mma_available(cc) || turing_mma_available(cc) || amd_wmma_available(cc);
        case LM_GGML_TYPE_BF16:
            return ampere_mma_available(cc) || amd_wmma_available(cc);
        default:
            return false;
    }
}
