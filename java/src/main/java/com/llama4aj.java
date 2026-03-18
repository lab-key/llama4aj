package com;

import java.util.function.Consumer;

/**
 * llama4aj: The ultra-concise Java identity for llama.cpp
 * 
 * Usage:
 * import com.llama4aj;
 * 
 * llama4aj.generate("model.gguf", "Hello!", System.out::print);
 */
public class llama4aj implements AutoCloseable {

    private long contextPtr;
    private static String loadedLib = "unknown";

    // --- Configuration ---

    public static class Config {
        public int n_ctx = 2048;
        public int n_gpu_layers = 0;
        public int n_batch = 512;
        public boolean use_mlock = true;
        public boolean use_mmap = true;
        public int n_threads = Runtime.getRuntime().availableProcessors();
        public boolean flash_attn = false;

        public Config nCtx(int n) { this.n_ctx = n; return this; }
        public Config gpuLayers(int n) { this.n_gpu_layers = n; return this; }
        public Config batchSize(int n) { this.n_batch = n; return this; }
        public Config threads(int n) { this.n_threads = n; return this; }
        public Config flashAttn(boolean b) { this.flash_attn = b; return this; }

        public String toJson() {
            return "{" +
                    "\"n_ctx\":" + n_ctx + "," +
                    "\"n_gpu_layers\":" + n_gpu_layers + "," +
                    "\"n_batch\":" + n_batch + "," +
                    "\"use_mlock\":" + use_mlock + "," +
                    "\"use_mmap\":" + use_mmap + "," +
                    "\"n_threads\":" + n_threads + "," +
                    "\"flash_attn\":" + flash_attn +
                    "}";
        }
    }

    // --- High-Level API ---

    public static void generate(String modelPath, String prompt, Consumer<String> onToken) {
        try (llama4aj model = load(modelPath)) {
            model.generate(prompt, onToken);
        }
    }

    public static llama4aj load(String modelPath) {
        return load(modelPath, new Config());
    }

    public static llama4aj load(String modelPath, Config config) {
        long ptr = nativeLoadModel(modelPath, config.toJson());
        if (ptr == 0) throw new RuntimeException("Failed to load model: " + modelPath);
        return new llama4aj(ptr);
    }

    public void generate(String prompt, Consumer<String> onToken) {
        String paramsJson = "{\"prompt\":\"" + prompt.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r") + "\", \"stream\": true}";
        nativeCompletion(contextPtr, paramsJson, (token, stop) -> onToken.accept(token));
    }

    // --- Lifecycle ---

    private llama4aj(long contextPtr) {
        this.contextPtr = contextPtr;
    }

    @Override
    public void close() {
        if (contextPtr != 0) {
            nativeDestroyContext(contextPtr);
            contextPtr = 0;
        }
    }

    public void interrupt() {
        if (contextPtr != 0) nativeInterrupt(contextPtr);
    }

    public static String getLoadedLibrary() {
        return loadedLib;
    }

    // --- Native Bridge ---

    private static native long nativeLoadModel(String modelPath, String configJson);
    private static native void nativeDestroyContext(long contextPtr);
    private static native void nativeCompletion(long contextPtr, String completionParamsJson, CompletionCallback callback);
    private static native void nativeInterrupt(long contextPtr);

    public interface CompletionCallback {
        void onTokenReceived(String token, boolean stop);
    }

    // --- Advanced API ---

    public void completion(String json, CompletionCallback callback) {
        nativeCompletion(contextPtr, json, callback);
    }

    // --- Legacy Support (Deprecated) ---
    // These allow existing code to work while migrating to the ultra-concise API.
    
    @Deprecated
    public static class LlamaConfig extends Config {}

    @Deprecated
    public static llama4aj create(String modelPath) {
        return load(modelPath);
    }

    @Deprecated
    public static llama4aj create(String modelPath, Config config) {
        return load(modelPath, config);
    }

    @Deprecated
    public void destroy() {
        close();
    }

    static {
        // Shared loading logic...
        boolean loadedDesktop = false;
        String os = System.getProperty("os.name").toLowerCase();

        if (os.contains("win") || os.contains("mac") || os.contains("nix") || os.contains("nux")) {
            String forcedVariant = System.getProperty("llama4aj.variant");
            if (forcedVariant != null && !forcedVariant.trim().isEmpty()) {
                try {
                    System.loadLibrary(forcedVariant);
                    loadedLib = forcedVariant;
                    loadedDesktop = true;
                } catch (UnsatisfiedLinkError e) {
                    System.err.println("llama4aj: Failed to load forced variant '" + forcedVariant + "'");
                }
            }

            if (!loadedDesktop) {
                String[] variants = { 
                    "ajllama_desktop_cuda", "ajllama_desktop_opencl", "ajllama_desktop_vulkan",
                    "ajllama_desktop_hip", "ajllama_desktop_amx", "ajllama_desktop_avx2",
                    "ajllama_desktop_x86_64", "ajllama_desktop_arm64", "ajllama_desktop_cpu",
                    "ajllama_desktop" 
                };
                for (String variant : variants) {
                    try {
                        System.loadLibrary(variant);
                        loadedLib = variant;
                        loadedDesktop = true;
                        break;
                    } catch (UnsatisfiedLinkError e) {}
                }
            }
        }

        if (!loadedDesktop) {
            String[] androidVariants = {
                "ajllama_jni_v8_2_dotprod_i8mm_hexagon_opencl",
                "ajllama_jni_v8_2_dotprod_i8mm",
                "ajllama_jni_v8_2_dotprod",
                "ajllama_jni_v8_2_i8mm",
                "ajllama_jni_v8_2",
                "ajllama_jni_v8",
                "ajllama_jni",
                "ajllama_jni_x86_64"
            };
            for (String variant : androidVariants) {
                try {
                    System.loadLibrary(variant);
                    loadedLib = variant;
                    break;
                } catch (UnsatisfiedLinkError e) {}
            }
        }
        
        try { System.loadLibrary("c++_shared"); } catch (UnsatisfiedLinkError e) {}
    }
}
