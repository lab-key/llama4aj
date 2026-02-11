#include <jni.h>
#include "llama.h"
#include "rn-llama.h"

extern "C" JNIEXPORT jlong JNICALL
Java_com_rnllama_LlamaContext_nativeLoadModel(JNIEnv *env, jclass /*clazz*/, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    auto *ctx = rn_llama_init_from_file(path);
    env->ReleaseStringUTFChars(model_path, path);
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_nativeDestroyContext(JNIEnv *env, jclass /*clazz*/, jlong context_ptr) {
    auto *ctx = reinterpret_cast<rn_llama_context *>(context_ptr);
    rn_llama_free(ctx);
}
