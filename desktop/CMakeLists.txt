cmake_minimum_required(VERSION 3.10)

project(ajllama_desktop)

option(HEXAGON_SDK_ROOT "Hexagon SDK root directory" OFF)
option(HEXAGON_TOOLS_ROOT "Hexagon SDK toolchain directory" OFF)

find_package(Python3 REQUIRED)

find_library(LOG_LIB log)

# Removed CMAKE_ANDROID_STL_TYPE as it's Android-specific for desktop build
set(CMAKE_CXX_STANDARD 17)

set(CPP_DIR ${CMAKE_SOURCE_DIR}/../cpp)

# --- Include Directories ---
include_directories(
    ${CPP_DIR}
    ${CPP_DIR}/common
    ${CPP_DIR}/common/jinja
    ${CPP_DIR}/ggml-cpu
    ${CPP_DIR}/tools/mtmd
)

# --- Source File Globbing ---
file(GLOB MODEL_FILES ${CPP_DIR}/models/*.cpp)
file(GLOB GGML_CPU_C_FILES ${CPP_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES ${CPP_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES ${CPP_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB LLAMA_FILES ${CPP_DIR}/llama*.cpp)
file(GLOB MTMD_FILES ${CPP_DIR}/tools/mtmd/*.cpp)
file(GLOB MTMD_MODEL_FILES ${CPP_DIR}/tools/mtmd/models/*.cpp)
file(GLOB COMMON_FILES ${CPP_DIR}/common/*.cpp)
file(GLOB JINJA_FILES ${CPP_DIR}/common/jinja/*.cpp)
# Explicitly include only ThreadPool.cpp from jsi, as other files depend on React Native JSI headers
set(JSI_FILES ${CPP_DIR}/jsi/ThreadPool.cpp)

# --- Main Source File List ---
set(
    AJLLAMA_SOURCE_FILES
    # JNI Wrapper
    ${CPP_DIR}/ajllamaJNI.cpp

    # Core GGML files
    ${CPP_DIR}/ggml.c
    ${CPP_DIR}/ggml-alloc.c
    ${CPP_DIR}/ggml-backend.cpp
#    ${CPP_DIR}/ggml-backend-dl.cpp
    ${CPP_DIR}/ggml-backend-reg.cpp
    ${CPP_DIR}/ggml-opt.cpp
    ${CPP_DIR}/ggml-threading.cpp
    ${CPP_DIR}/ggml-quants.c
    ${CPP_DIR}/gguf.cpp

    # GGML CPU files (generic)
    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}

    # Llama files
    ${LLAMA_FILES}
    ${CPP_DIR}/unicode-data.cpp
    ${CPP_DIR}/unicode.cpp

    # Common utilities
    ${COMMON_FILES}
    ${CPP_DIR}/anyascii.c

    # Jinja template engine
    ${JINJA_FILES}

    # JSI files (including ThreadPool)
    ${JSI_FILES}

    # Multimodal support
    ${MTMD_MODEL_FILES}
    ${MTMD_FILES}

    # Core "rn" logic files we are reusing
    ${CPP_DIR}/rn-llama.cpp
    ${CPP_DIR}/rn-completion.cpp
    ${CPP_DIR}/rn-tts.cpp
    ${CPP_DIR}/rn-slot.cpp
    ${CPP_DIR}/rn-slot-manager.cpp

    # Model implementations
    ${MODEL_FILES}
)

# --- Library Definition Function ---
function(build_ajllama_jni jni_name arch cpu_flags) # Added arch parameter back for consistency with Android
    # Remove ENABLE_OPENCL and ENABLE_HEXAGON from function parameters
    # as they will be handled by top-level options.

    if (NOT "${arch}" STREQUAL "generic")
        set(SOURCE_FILES_ARCH
            ${CPP_DIR}/ggml-cpu/arch/${arch}/quants.c
            ${CPP_DIR}/ggml-cpu/arch/${arch}/repack.cpp
        )
    endif ()

    add_library(
        ${jni_name}
        SHARED
        ${AJLLAMA_SOURCE_FILES}
        ${SOURCE_FILES_ARCH}
    )

    # Link Libraries
    find_package(JNI REQUIRED) # Keeping JNI as per user's provided file
    target_link_libraries(${jni_name} PRIVATE
        ${JNI_LIBRARIES}
        pthread
    )
    target_include_directories(${jni_name} PRIVATE ${JNI_INCLUDE_DIRS})

    target_compile_definitions(${jni_name} PRIVATE
        RNLLAMA_VERBOSE=1
    )

    # Compile Options
    target_compile_options(${jni_name} PRIVATE
        -DLM_GGML_USE_CPU
        -pthread
        -ffast-math
        -flto
        -D_GNU_SOURCE # Keep for Linux compatibility, can be conditional for Windows/macOS if needed
        ${cpu_flags}
        $<$<STREQUAL:"${arch}","x86">:-DLM_GGML_CPU_HAS_AVX> # Add AVX support for x86
        )
        
        target_compile_options(${jni_name} PRIVATE -O3 -DNDEBUG)
        target_compile_options(${jni_name} PRIVATE -DRNLLAMA_VERBOSE=0) # Aligns with working Android build
        target_compile_options(${jni_name} PRIVATE
            -fPIC
            -fno-finite-math-only
        )
    target_compile_options(${jni_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${jni_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${jni_name} PRIVATE -flto)

    # CUDA backend support
    if (LM_GGML_USE_CUDA)
        message(STATUS "Building with CUDA support for ${jni_name}")
        find_package(CUDA REQUIRED)
        target_link_libraries(${jni_name} PRIVATE CUDA::cudart)
        target_compile_options(${jni_name} PRIVATE -DLM_GGML_USE_CUDA)
        # Add CUDA source files
        file(GLOB GGML_CUDA_FILES ${CPP_DIR}/ggml-cuda/*.cu ${CPP_DIR}/ggml-cuda/*.cpp)
        target_sources(${jni_name} PRIVATE ${GGML_CUDA_FILES})
    endif ()

    # Metal backend support (macOS)
    if (APPLE)
        message(STATUS "Building with Metal support for ${jni_name}")
        target_link_libraries(${jni_name} PRIVATE "-framework Foundation" "-framework Metal" "-framework MetalKit")
        target_compile_options(${jni_name} PRIVATE -DLM_GGML_USE_METAL)
        
        # Add Metal source files
        file(GLOB GGML_METAL_FILES ${CPP_DIR}/ggml-metal/*.m ${CPP_DIR}/ggml-metal/*.cpp)
        target_sources(${jni_name} PRIVATE ${GGML_METAL_FILES})
        
        # Set source properties for Objective-C if needed
        set_source_files_properties(${CPP_DIR}/ggml-metal/ggml-metal.m PROPERTIES COMPILE_FLAGS "-fno-objc-arc")
    endif()

    # OpenCL backend support
    if (LM_GGML_USE_OPENCL)
        message(STATUS "Building with OpenCL support for ${jni_name}")
        find_package(OpenCL REQUIRED)
        target_link_libraries(${jni_name} PRIVATE OpenCL::OpenCL)
        target_compile_options(${jni_name} PRIVATE -DLM_GGML_USE_OPENCL)
        # Add OpenCL source files
        target_sources(${jni_name} PRIVATE ${CPP_DIR}/ggml-opencl/ggml-opencl.cpp)
    endif ()

    # Vulkan backend support
    if (LM_GGML_USE_VULKAN)
        message(STATUS "Building with Vulkan support for ${jni_name}")
        find_package(Vulkan REQUIRED)
        target_link_libraries(${jni_name} PRIVATE Vulkan::Vulkan)
        target_compile_options(${jni_name} PRIVATE -DLM_GGML_USE_VULKAN)
        file(GLOB GGML_VULKAN_FILES ${CPP_DIR}/ggml-vulkan/*.cpp)
        target_sources(${jni_name} PRIVATE ${GGML_VULKAN_FILES})
    endif()

    # ROCm/HIP backend support
    if (LM_GGML_USE_HIP)
        message(STATUS "Building with ROCm/HIP support for ${jni_name}")
        find_package(HIP REQUIRED)
        target_link_libraries(${jni_name} PRIVATE HIP::HIP)
        target_compile_options(${jni_name} PRIVATE -DLM_GGML_USE_HIP)
        file(GLOB GGML_HIP_FILES ${CPP_DIR}/ggml-hip/*.cpp)
        target_sources(${jni_name} PRIVATE ${GGML_HIP_FILES})
    endif()

endfunction()

# --- Helper function to check if a variant should be built ---
function(should_build_variant variant_name result_var)
    # AJLLAMA_VARIANT can be:
    # - Empty string ("") = build all variants
    # - Specific variant name = build only that variant
    # Valid variant names: "generic", "x86_64", "cuda", "opencl", "vulkan", "hip"

    if (NOT DEFINED AJLLAMA_VARIANT OR AJLLAMA_VARIANT STREQUAL "")
        # Build all variants
        set(${result_var} TRUE PARENT_SCOPE)
    elseif (AJLLAMA_VARIANT STREQUAL variant_name)
        # Build only this specific variant
        set(${result_var} TRUE PARENT_SCOPE)
    else()
        # Skip this variant
        set(${result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# --- Build Targets ---
option(LM_GGML_USE_CUDA "Enable CUDA backend" OFF)
option(LM_GGML_USE_OPENCL "Enable OpenCL backend" OFF)
option(LM_GGML_USE_VULKAN "Enable Vulkan backend" OFF)
option(LM_GGML_USE_HIP "Enable ROCm/HIP backend" OFF)

# Build a generic CPU version
should_build_variant("generic" BUILD_GENERIC)
if (BUILD_GENERIC)
    build_ajllama_jni("ajllama_desktop_cpu" "generic" "")
endif()

# Build for x86_64 if the host system is x86_64
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    should_build_variant("x86_64" BUILD_X86_64)
    if (BUILD_X86_64)
        build_ajllama_jni("ajllama_desktop_x86_64" "x86" "-march=x86-64;-mtune=generic;-msse4.2;-mpopcnt")
    endif()

    should_build_variant("avx2" BUILD_AVX2)
    if (BUILD_AVX2)
        # AVX2 + FMA + F16C
        build_ajllama_jni("ajllama_desktop_avx2" "x86" "-march=x86-64-v3")
    endif()

    should_build_variant("amx" BUILD_AMX)
    if (BUILD_AMX)
        # AMX support (requires more recent compilers/CPUs)
        build_ajllama_jni("ajllama_desktop_amx" "x86" "-march=x86-64-v4;-mamx-tile;-mamx-int8;-mamx-bf16")
    endif()
endif()

# Build for arm64 if the host system is arm64 (mostly for macos/linux arm)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    should_build_variant("arm64" BUILD_ARM64)
    if (BUILD_ARM64)
        build_ajllama_jni("ajllama_desktop_arm64" "arm" "-march=armv8-a")
    endif()
endif()

# Build targets for hardware acceleration backends if their options are ON
if (LM_GGML_USE_CUDA)
    should_build_variant("cuda" BUILD_CUDA)
    if (BUILD_CUDA)
        build_ajllama_jni("ajllama_desktop_cuda" "generic" "")
    endif()
endif()

if (LM_GGML_USE_OPENCL)
    should_build_variant("opencl" BUILD_OPENCL)
    if (BUILD_OPENCL)
        build_ajllama_jni("ajllama_desktop_opencl" "generic" "")
    endif()
endif()

if (LM_GGML_USE_VULKAN)
    should_build_variant("vulkan" BUILD_VULKAN)
    if (BUILD_VULKAN)
        build_ajllama_jni("ajllama_desktop_vulkan" "generic" "")
    endif()
endif()

if (LM_GGML_USE_HIP)
    should_build_variant("hip" BUILD_HIP)
    if (BUILD_HIP)
        build_ajllama_jni("ajllama_desktop_hip" "generic" "")
    endif()
endif()
