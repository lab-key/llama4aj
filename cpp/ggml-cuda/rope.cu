#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne0,
                                 const int            ne1,
                                 const int            s1,
                                 const int            s2,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    int       idst = row_dst * ne0 + i0;
    const int ix   = channel_x*s2 + row_x*s1 + i0;

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = row_x * ne0 + i0;
        idst += row_indices[channel_x] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            lm_ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            lm_ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 >= n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const float theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne0,
                                 const int            ne1,
                                 const int            s1,
                                 const int            s2,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    int       idst = row_dst * ne0 + i0 / 2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = row_x * ne0 + i0 / 2;
        idst += row_indices[channel_x] * set_rows_stride;
    }

    if (i0 >= n_dims) {
        dst[idst + i0 / 2 + 0] = lm_ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = lm_ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const float theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]          = lm_ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_dims / 2] = lm_ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_multi(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2,
        const int n_dims, const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors, const mrope_sections sections, const bool is_imrope) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0/2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    if (i0 >= n_dims) {
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];

        return;
    }

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) { // h
            theta_base = pos[channel_x + ne2 * 1]*powf(theta_scale, i0/2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) { // w
            theta_base = pos[channel_x + ne2 * 2]*powf(theta_scale, i0/2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) { // t
            theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);
        } else {
            theta_base = pos[channel_x + ne2 * 3]*powf(theta_scale, i0/2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);
        }
        else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[channel_x + ne2 * 1]*powf(theta_scale, i0/2.0f);
        }
        else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[channel_x + ne2 * 2]*powf(theta_scale, i0/2.0f);
        }
        else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[channel_x + ne2 * 3]*powf(theta_scale, i0/2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_vision(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims,
        const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
        const float theta_scale, const float * freq_factors, const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0/2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base = pos[channel_x]*powf(theta_scale, p);
    }
    else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base = pos[channel_x + ne2]*powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne0,
                           const int            ne1,
                           const int            s1,
                           const int            s2,
                           const int            n_dims,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           cudaStream_t         stream) {
    LM_GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
            freq_factors, row_indices, set_rows_stride);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
            freq_factors, row_indices, set_rows_stride);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne0,
                           const int            ne1,
                           const int            s1,
                           const int            s2,
                           const int            n_dims,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           cudaStream_t         stream) {
    LM_GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_neox<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
            freq_factors, row_indices, set_rows_stride);
    } else {
        rope_neox<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
            freq_factors, row_indices, set_rows_stride);
    }
}

template<bool forward, typename T>
static void rope_multi_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, const mrope_sections sections, const bool is_imrope, cudaStream_t stream) {
    LM_GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_multi<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
    } else {
        rope_multi<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
    }
}

template<bool forward, typename T>
static void rope_vision_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, const mrope_sections sections, cudaStream_t stream) {
    LM_GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void lm_ggml_cuda_op_rope_impl(lm_ggml_backend_cuda_context & ctx,
                            lm_ggml_tensor *               dst,
                            const lm_ggml_tensor *         set_rows = nullptr) {
    const lm_ggml_tensor * src0 = dst->src[0];
    const lm_ggml_tensor * src1 = dst->src[1];
    const lm_ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    lm_ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        LM_GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / lm_ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32 || src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32 ||  dst->type == LM_GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    LM_GGML_ASSERT(src0->type == dst->type || (src0->type == LM_GGML_TYPE_F32 && dst->type == LM_GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = lm_ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / lm_ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / lm_ggml_type_size(src0->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    mrope_sections sections;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & LM_GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & LM_GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == LM_GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == LM_GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        LM_GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        LM_GGML_ASSERT(n_dims == ne00/2);
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    lm_ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == LM_GGML_TYPE_F32 && dst_type == LM_GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, s01, s02, n_dims,
                                                  nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
                                                  freq_factors, row_indices, set_rows_stride, stream);
        } else if (src0->type == LM_GGML_TYPE_F32 && dst_type == LM_GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, s01, s02, n_dims,
                                                 nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
                                                 freq_factors, row_indices, set_rows_stride, stream);
        } else if (src0->type == LM_GGML_TYPE_F16 && dst_type == LM_GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, s01, s02, n_dims, nr,
                                                pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
                                                freq_factors, row_indices, set_rows_stride, stream);
        } else {
            LM_GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == LM_GGML_TYPE_F32) {
            rope_multi_cuda<forward>(
                (const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope, stream);
        } else if (src0->type == LM_GGML_TYPE_F16) {
            rope_multi_cuda<forward>(
                (const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope, stream);
        } else {
            LM_GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == LM_GGML_TYPE_F32) {
            rope_vision_cuda<forward>(
                (const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections, stream);
        } else if (src0->type == LM_GGML_TYPE_F16) {
            rope_vision_cuda<forward>(
                (const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, n_dims, nr, pos, freq_scale,
                freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections, stream);
        } else {
            LM_GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == LM_GGML_TYPE_F32 && dst_type == LM_GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, s01, s02, n_dims,
                                                  nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
                                                  freq_factors, row_indices, set_rows_stride, stream);
        } else if (src0->type == LM_GGML_TYPE_F32 && dst_type == LM_GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, s01, s02, n_dims,
                                                 nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
                                                 freq_factors, row_indices, set_rows_stride, stream);
        } else if (src0->type == LM_GGML_TYPE_F16 && dst_type == LM_GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, s01, s02, n_dims, nr,
                                                pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
                                                freq_factors, row_indices, set_rows_stride, stream);
        } else {
            LM_GGML_ABORT("fatal error");
        }
    }
}

void lm_ggml_cuda_op_rope(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void lm_ggml_cuda_op_rope_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void lm_ggml_cuda_op_rope_fused(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * rope, lm_ggml_tensor * set_rows) {
    lm_ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}
