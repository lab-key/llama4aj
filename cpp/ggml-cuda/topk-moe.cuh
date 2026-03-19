#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

void lm_ggml_cuda_op_topk_moe(lm_ggml_backend_cuda_context & ctx,
                           const lm_ggml_tensor *         logits,
                           lm_ggml_tensor *               weights,
                           lm_ggml_tensor *               ids,
                           const bool                  with_norm,
                           const bool                  delayed_softmax = false,
                           lm_ggml_tensor *               weight_clamp    = nullptr);

bool lm_ggml_cuda_should_use_topk_moe(const lm_ggml_tensor * softmax,
                                   const lm_ggml_tensor * weights,
                                   const lm_ggml_tensor * get_rows,
                                   const lm_ggml_tensor * argsort,
                                   const lm_ggml_tensor * clamp,
                                   int n_expert);

std::initializer_list<enum lm_ggml_op> lm_ggml_cuda_topk_moe_ops(bool with_norm, bool delayed_softmax = false);
