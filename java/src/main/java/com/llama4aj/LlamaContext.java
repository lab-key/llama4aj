package com.llama4aj;

public class LlamaContext {

    private long contextPtr;
    private static String loadedLib = "unknown";

    private LlamaContext(long contextPtr) {
        this.contextPtr = contextPtr;
    }

    public static String getLoadedLibrary() {
        return loadedLib;
    }

    public static LlamaContext create(String modelPath) {
        long contextPtr = nativeLoadModel(modelPath);
        if (contextPtr == 0) {
            return null;
        }
        return new LlamaContext(contextPtr);
    }

    public void destroy() {
        if (contextPtr != 0) {
            nativeDestroyContext(contextPtr);
            contextPtr = 0;
        }
    }

    public void interrupt() {
        if (contextPtr != 0) {
            nativeInterrupt(contextPtr);
        }
    }

    public interface CompletionCallback {
        void onTokenReceived(String tokenDataJson);
    }

    public void completion(String completionParamsJson, CompletionCallback callback) {
        if (contextPtr == 0) {
            return;
        }
        nativeCompletion(contextPtr, completionParamsJson, callback);
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            destroy();
        } finally {
            super.finalize();
        }
    }

    // Native methods implemented in allamaJNI.cpp
    private static native long nativeLoadModel(String modelPath);
    private static native void nativeDestroyContext(long contextPtr);
    private static native void nativeCompletion(long contextPtr, String completionParamsJson, CompletionCallback callback);
    private static native void nativeInterrupt(long contextPtr);

    // Static block to load the native library
    static {
        // Try to load the most optimized library variant first
        // If it fails (UnsatisfiedLinkError due to missing symbols/instructions), fall back to the next best one
        
        // 1. Full feature set: v8.2 + dotprod + i8mm + Hexagon/OpenCL (if built)
        try {
            System.loadLibrary("ajllama_jni_v8_2_dotprod_i8mm_hexagon_opencl");
            loadedLib = "ajllama_jni_v8_2_dotprod_i8mm_hexagon_opencl";
        } catch (UnsatisfiedLinkError e1) {
            try {
                // 2. High performance: v8.2 + dotprod + i8mm
                System.loadLibrary("ajllama_jni_v8_2_dotprod_i8mm");
                loadedLib = "ajllama_jni_v8_2_dotprod_i8mm";
            } catch (UnsatisfiedLinkError e2) {
                try {
                    // 3. Dotprod only
                    System.loadLibrary("ajllama_jni_v8_2_dotprod");
                    loadedLib = "ajllama_jni_v8_2_dotprod";
                } catch (UnsatisfiedLinkError e3) {
                    try {
                        // 4. i8mm only
                        System.loadLibrary("ajllama_jni_v8_2_i8mm");
                        loadedLib = "ajllama_jni_v8_2_i8mm";
                    } catch (UnsatisfiedLinkError e4) {
                        try {
                            // 5. v8.2 base
                            System.loadLibrary("ajllama_jni_v8_2");
                            loadedLib = "ajllama_jni_v8_2";
                        } catch (UnsatisfiedLinkError e5) {
                            try {
                                // 6. v8 base (Standard Arm64)
                                System.loadLibrary("ajllama_jni_v8");
                                loadedLib = "ajllama_jni_v8";
                            } catch (UnsatisfiedLinkError e6) {
                                try {
                                    // 7. Generic fallback / x86_64
                                    System.loadLibrary("ajllama_jni");
                                    loadedLib = "ajllama_jni";
                                } catch (UnsatisfiedLinkError e7) {
                                     // 8. Legacy/Original name fallback
                                     System.loadLibrary("ajllama_jni_x86_64");
                                     loadedLib = "ajllama_jni_x86_64";
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Also load c++_shared if not already loaded by the system
        try {
            System.loadLibrary("c++_shared");
        } catch (UnsatisfiedLinkError e) {
            // Might be already loaded
        }
    }
}
