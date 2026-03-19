#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef LM_GGML_USE_HIP
#define LM_GGML_CUDA_NAME "ROCm"
#define LM_GGML_CUBLAS_NAME "hipBLAS"
#elif defined(LM_GGML_USE_MUSA)
#define LM_GGML_CUDA_NAME "MUSA"
#define LM_GGML_CUBLAS_NAME "muBLAS"
#else
#define LM_GGML_CUDA_NAME "CUDA"
#define LM_GGML_CUBLAS_NAME "cuBLAS"
#endif
#define LM_GGML_CUDA_MAX_DEVICES       16

// backend API
LM_GGML_BACKEND_API lm_ggml_backend_t lm_ggml_backend_cuda_init(int device);

LM_GGML_BACKEND_API bool lm_ggml_backend_is_cuda(lm_ggml_backend_t backend);

// device buffer
LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_cuda_host_buffer_type(void);

LM_GGML_BACKEND_API int  lm_ggml_backend_cuda_get_device_count(void);
LM_GGML_BACKEND_API void lm_ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
LM_GGML_BACKEND_API void lm_ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

LM_GGML_BACKEND_API bool lm_ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
LM_GGML_BACKEND_API void lm_ggml_backend_cuda_unregister_host_buffer(void * buffer);

LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
