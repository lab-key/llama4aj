#include <jni.h>
#include <string>
#include <vector>
#include <thread>
#ifdef __ANDROID__
#include <android/log.h>
#else
#include <stdio.h>
#include <stdarg.h>
#endif

#include "rn-llama.h"
#include "rn-completion.h"
#include "nlohmann/json.hpp"
#include "jsi/ThreadPool.h" // Added for ThreadPool initialization

// --- Logging ---
static void jni_log(const char *format, ...) {
#ifdef __ANDROID__
    va_list args;
    va_start(args, format);
    __android_log_vprint(ANDROID_LOG_DEBUG, "ajllama_jni", format, args);
    va_end(args);
#else
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    printf("\n");
    va_end(args);
    fflush(stdout);
#endif
}

// --- JNI Helper Structs ---

// Global state to hold the JavaVM pointer
static JavaVM* g_jvm = nullptr;

// Data passed to the C++ completion callback
struct JniCallbackContext {
    jobject callback_obj; // Global reference to the Java callback
};

// --- JNI OnLoad ---

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_jvm = vm;
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    jni_log("JNI_OnLoad successful");
    return JNI_VERSION_1_6;
}

// --- Completion Callback (from C++ to Java) ---

void completion_callback_c(const char* token_data_json, void* user_data) {
    JNIEnv* env = nullptr;
    bool attached = false;

    // Attach the current C++ thread to the JVM to get a valid JNIEnv
    if (g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
#ifdef __ANDROID__
        if (g_jvm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
#else
        if (g_jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) != JNI_OK) {
#endif
            jni_log("ERROR: Failed to attach current thread to JVM");
            return;
        }
        attached = true;
    }

    JniCallbackContext* context = static_cast<JniCallbackContext*>(user_data);
    jobject callback_obj = context->callback_obj;

    jclass callback_class = env->GetObjectClass(callback_obj);
    if (callback_class == nullptr) {
        jni_log("ERROR: Could not find Java callback class");
        if (attached) g_jvm->DetachCurrentThread();
        return;
    }

    jmethodID on_token_method = env->GetMethodID(callback_class, "onTokenReceived", "(Ljava/lang/String;)V");
    if (on_token_method == nullptr) {
        jni_log("ERROR: Could not find onTokenReceived method on callback object");
        env->DeleteLocalRef(callback_class);
        if (attached) g_jvm->DetachCurrentThread();
        return;
    }

    // Create a Java string from the C++ string
    jstring java_token_data = env->NewStringUTF(token_data_json);
    if (java_token_data == nullptr) {
        jni_log("ERROR: Failed to create Java string from token data");
        env->DeleteLocalRef(callback_class);
        if (attached) g_jvm->DetachCurrentThread();
        return;
    }

    // Call the Java method
    env->CallVoidMethod(callback_obj, on_token_method, java_token_data);

    // Clean up local references
    env->DeleteLocalRef(java_token_data);
    env->DeleteLocalRef(callback_class);

    // Detach the thread if it was attached
    if (attached) {
        g_jvm->DetachCurrentThread();
    }
}


// --- Native Method Implementations ---

extern "C" JNIEXPORT jlong JNICALL
Java_com_llama4aj_nativeLoadModel(JNIEnv *env, jclass /*clazz*/, jstring model_path_j, jstring config_json_j) {
    jni_log("=== nativeLoadModel START ===");
    try {
        ThreadPool::getInstance().ensureRunning();
        jni_log("ThreadPool running");
    } catch (const std::exception& e) {
        jni_log("ERROR: ThreadPool init failed: %s", e.what());
    }

    if (model_path_j == nullptr) {
        jni_log("ERROR: model_path is null");
        return 0;
    }

    const char *model_path_c = env->GetStringUTFChars(model_path_j, nullptr);
    if (model_path_c == nullptr) {
        jni_log("ERROR: Could not get UTF chars from model_path jstring");
        return 0;
    }

    const char *config_json_c = nullptr;
    if (config_json_j != nullptr) {
        config_json_c = env->GetStringUTFChars(config_json_j, nullptr);
    }

    jni_log("Preparing params for: %s", model_path_c);
    
    // Initialize defaults
    ::common_params params;
    params.model.path = model_path_c;
    params.n_ctx = 2048;
    params.n_gpu_layers = 0;
    params.use_mlock = true;
    params.n_batch = 512;
    params.cpuparams.n_threads = std::thread::hardware_concurrency();
    params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();

    // Parse JSON config if provided
    if (config_json_c != nullptr) {
        try {
            auto config = nlohmann::json::parse(config_json_c);
            jni_log("Parsing config JSON: %s", config_json_c);
            
            if (config.contains("n_ctx")) params.n_ctx = config["n_ctx"].get<int>();
            if (config.contains("n_gpu_layers")) params.n_gpu_layers = config["n_gpu_layers"].get<int>();
            if (config.contains("n_batch")) params.n_batch = config["n_batch"].get<int>();
            if (config.contains("use_mlock")) params.use_mlock = config["use_mlock"].get<bool>();
            if (config.contains("use_mmap")) params.use_mmap = config["use_mmap"].get<bool>();
            
            if (config.contains("n_threads")) {
                int threads = config["n_threads"].get<int>();
                params.cpuparams.n_threads = threads;
                params.cpuparams_batch.n_threads = threads;
            }

            if (config.contains("flash_attn")) {
                params.flash_attn_type = config["flash_attn"].get<bool>() ? 
                    LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            }
            
            jni_log("Params updated from JSON: n_gpu_layers=%d, n_ctx=%d", params.n_gpu_layers, params.n_ctx);
        } catch (const std::exception& e) {
            jni_log("WARNING: Failed to parse config JSON: %s. Using defaults.", e.what());
        }
        env->ReleaseStringUTFChars(config_json_j, config_json_c);
    }

    rnllama::llama_rn_context* ctx = nullptr;
    try {
        ctx = new rnllama::llama_rn_context();
        jni_log("rnllama context created, loading model...");
        if (!ctx->loadModel(params)) {
            jni_log("ERROR: rn-llama context failed to load model.");
            delete ctx;
            env->ReleaseStringUTFChars(model_path_j, model_path_c);
            return 0;
        }
        jni_log("Model loaded successfully, attaching threadpools...");
        ctx->attachThreadpoolsIfAvailable();
        jni_log("Threadpools attached (if available)");
    } catch (const std::exception& e) {
        jni_log("EXCEPTION during model load: %s", e.what());
        if (ctx) delete ctx;
        env->ReleaseStringUTFChars(model_path_j, model_path_c);
        return 0;
    }

    env->ReleaseStringUTFChars(model_path_j, model_path_c);
    jni_log("=== nativeLoadModel SUCCESS, context ptr: %p ===", ctx);
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_com_llama4aj_nativeDestroyContext(JNIEnv *env, jclass /*clazz*/, jlong context_ptr) {
    if (context_ptr == 0) return;
    jni_log("Destroying context: %p", (void*)context_ptr);
    rnllama::llama_rn_context* ctx = reinterpret_cast<rnllama::llama_rn_context*>(context_ptr);
    delete ctx;
}

extern "C" JNIEXPORT void JNICALL
Java_com_llama4aj_nativeCompletion(JNIEnv *env, jclass /*clazz*/, jlong context_ptr, jstring completion_params_json_j, jobject callback_obj_j) {
    jni_log("--- nativeCompletion START ---");
    if (context_ptr == 0) {
        jni_log("ERROR: Context pointer is null");
        return;
    }
    if (completion_params_json_j == nullptr || callback_obj_j == nullptr) {
        jni_log("ERROR: JSON params or callback object is null");
        return;
    }

    rnllama::llama_rn_context* ctx = reinterpret_cast<rnllama::llama_rn_context*>(context_ptr);
    const char *params_c = env->GetStringUTFChars(completion_params_json_j, nullptr);
    std::string params_str(params_c);
    env->ReleaseStringUTFChars(completion_params_json_j, params_c);

    jobject callback_global_ref = env->NewGlobalRef(callback_obj_j);
    if (callback_global_ref == nullptr) {
        jni_log("ERROR: Failed to create global ref for callback object");
        return;
    }

    JniCallbackContext* callback_context = new JniCallbackContext{callback_global_ref};

    std::thread completion_thread([ctx, params_str, callback_context]() {
        try {
            ctx->completion->rewind();

            nlohmann::json j_params = nlohmann::json::parse(params_str);
            ctx->params.prompt = j_params.value("prompt", "");
            ctx->params.sampling.temp = j_params.value("temperature", 0.8f);
            
            if (!ctx->completion->initSampling()) {
                throw std::runtime_error("Failed to init sampling");
            }

            ctx->completion->loadPrompt({});

            if (ctx->completion->context_full) {
                 throw std::runtime_error("Context is full");
            }

            ctx->completion->beginCompletion(::COMMON_CHAT_FORMAT_CONTENT_ONLY, ::COMMON_REASONING_FORMAT_NONE, false);

            while (ctx->completion->has_next_token) {
                if (ctx->completion->is_interrupted) {
                    jni_log("Completion interrupted.");
                    break;
                }
                rnllama::completion_token_output token_output = ctx->completion->doCompletion();

                nlohmann::json token_data_json;
                token_data_json["content"] = rnllama::tokens_to_output_formatted_string(ctx->ctx, token_output.tok);
                token_data_json["stop"] = !ctx->completion->has_next_token || ctx->completion->stopped_word;

                completion_callback_c(token_data_json.dump().c_str(), callback_context);

                if (ctx->completion->stopped_word) {
                    break;
                }
            }
            ctx->completion->endCompletion();
            jni_log("Completion thread finished successfully.");

        } catch (const std::exception& e) {
            jni_log("EXCEPTION in completion thread: %s", e.what());
        }

        // --- Cleanup ---
        JNIEnv* cleanup_env = nullptr;
        if (g_jvm->GetEnv(reinterpret_cast<void**>(&cleanup_env), JNI_VERSION_1_6) != JNI_OK) {
#ifdef __ANDROID__
            if (g_jvm->AttachCurrentThread(&cleanup_env, nullptr) == JNI_OK) {
#else
            if (g_jvm->AttachCurrentThread(reinterpret_cast<void**>(&cleanup_env), nullptr) == JNI_OK) {
#endif
                cleanup_env->DeleteGlobalRef(callback_context->callback_obj);
                g_jvm->DetachCurrentThread();
            }
        } else {
            cleanup_env->DeleteGlobalRef(callback_context->callback_obj);
        }
        delete callback_context;
    });

    completion_thread.detach();
    jni_log("--- nativeCompletion END (thread detached) ---");
}

extern "C" JNIEXPORT void JNICALL
Java_com_llama4aj_nativeInterrupt(JNIEnv */*env*/, jclass /*clazz*/, jlong context_ptr) {
    jni_log("nativeInterrupt, context ptr: %p", (void*)context_ptr);
    if (context_ptr == 0) return;
    auto* ctx = reinterpret_cast<rnllama::llama_rn_context*>(context_ptr);
    if (ctx->completion) {
        ctx->completion->is_interrupted = true;
        jni_log("Interruption signal sent");
    }
}