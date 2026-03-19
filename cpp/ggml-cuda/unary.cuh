#pragma once
#include "common.cuh"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void lm_ggml_cuda_op_abs(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_sgn(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_neg(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_step(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_gelu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_silu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_silu_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_gelu_erf(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_gelu_quick(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_tanh(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_relu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_sigmoid(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_hardsigmoid(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_exp(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_hardswish(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_leaky_relu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_sqr(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_sqrt(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_sin(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_cos(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_log(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_expm1(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_softplus(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_elu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_floor(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_ceil(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_round(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_trunc(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_reglu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_geglu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_swiglu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_swiglu_oai(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_geglu_erf(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_geglu_quick(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

void lm_ggml_cuda_op_xielu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst);

__device__ __forceinline__ float lm_ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float lm_ggml_cuda_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float lm_ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}
