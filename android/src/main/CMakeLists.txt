cmake_minimum_required(VERSION 3.10)

project(ajllama)

option(HEXAGON_SDK_ROOT "Hexagon SDK root directory" OFF)
option(HEXAGON_TOOLS_ROOT "Hexagon SDK toolchain directory" OFF)

find_package(Python3 REQUIRED)

find_library(LOG_LIB log)

set(CMAKE_ANDROID_STL_TYPE c++_shared)
set(CMAKE_CXX_STANDARD 17)

set(CPP_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

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
    ${CMAKE_SOURCE_DIR}/allamaJNI.cpp

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
function(build_ajllama_jni jni_name arch cpu_flags)
    set(ENABLE_OPENCL OFF)
    if (${jni_name} MATCHES ".*_opencl$")
        set(ENABLE_OPENCL OFF)
    endif ()

    set(ENABLE_HEXAGON OFF)
    if (${jni_name} MATCHES ".*_hexagon.*")
        set(ENABLE_HEXAGON OFF)
    endif ()

    # Architecture-specific sources
    set(AJLLAMA_ARCH_SOURCE_FILES "")
    if (NOT ${arch} STREQUAL "generic")
        if (ANDROID_ABI STREQUAL "arm64-v8a" OR ANDROID_ABI STREQUAL "x86_64")
            set(AJLLAMA_ARCH_SOURCE_FILES
                ${CPP_DIR}/ggml-cpu/arch/${arch}/quants.c
                ${CPP_DIR}/ggml-cpu/arch/${arch}/repack.cpp
            )
        endif()
    endif()

    add_library(
        ${jni_name}
        SHARED
        ${AJLLAMA_SOURCE_FILES}
        ${AJLLAMA_ARCH_SOURCE_FILES}
    )

    # Link Libraries
    target_link_libraries(${jni_name} PRIVATE
        ${LOG_LIB}
        android
    )

    # Compile Options
    target_compile_options(${jni_name} PRIVATE
        -DLM_GGML_USE_CPU
        -DLM_GGML_USE_CPU_REPACK
        -pthread
        -fvectorize
        -ffp-model=fast
        -fno-finite-math-only
        -flto
        -D_GNU_SOURCE
        ${cpu_flags}
    )

    if (${arch} STREQUAL "generic")
        target_compile_options(${jni_name} PRIVATE -DLM_GGML_CPU_GENERIC)
    endif ()

    if (CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${jni_name} PRIVATE -DAJLLAMA_ANDROID_ENABLE_LOGGING)
    endif ()

    target_compile_options(${jni_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${jni_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${jni_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${jni_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${jni_name} PRIVATE -flto)

    # OpenCL only for the special target
    if (ENABLE_OPENCL)
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/OpenCL-Headers)

        set(GGML_OPENCL_KERNELS
            add
            add_id
            argsort
            tri
            fill
            clamp
            cpy
            cvt
            diag_mask_inf
            div
            gelu
            gemv_noshuffle_general
            gemv_noshuffle
            get_rows
            glu
            group_norm
            solve_tri
            im2col_f32
            im2col_f16
            mean
            mul_mat_Ab_Bi_8x4
            mul_mv_f16_f16
            mul_mv_f16_f32_1row
            mul_mv_f16_f32_l4
            mul_mv_f16_f32
            mul_mv_f32_f32
            mul_mv_q4_0_f32
            mul_mv_q4_0_f32_v
            mul_mv_q4_0_f32_8x_flat
            mul_mv_q4_0_f32_1d_8x_flat
            mul_mv_q4_0_f32_1d_16x_flat
            mul_mv_q6_k_f32
            mul_mv_q6_k_f32_flat
            mul_mv_q8_0_f32
            mul_mv_q8_0_f32_flat
            mul_mv_mxfp4_f32
            mul_mv_mxfp4_f32_flat
            mul_mv_id_q4_0_f32_8x_flat
            mul_mv_id_q8_0_f32
            mul_mv_id_q8_0_f32_flat
            mul_mv_id_mxfp4_f32
            mul_mv_id_mxfp4_f32_flat
            gemm_moe_mxfp4_f32
            gemv_moe_mxfp4_f32
            mul_mm_f32_f32_l4_lm
            mul_mm_f16_f32_l4_lm
            mul_mm_q8_0_f32_l4_lm
            mul_mm_q8_0_f32_8x4
            gemv_noshuffle_general_q8_0_f32
            mul
            norm
            relu
            rms_norm
            rope
            scale
            set_rows
            sigmoid
            silu
            softmax_4_f32
            softmax_4_f16
            softmax_f32
            softmax_f16
            sqr
            sqrt
            ssm_conv
            sub
            sum_rows
            transpose
            concat
            tsembd
            upscale
            tanh
            expm1
            softplus
            pad
            repeat
            mul_mat_f16_f32
            mul_mm_f16_f32_kq_kqv
            conv2d
            conv2d_f16_f32
            flash_attn_f32_f16
            flash_attn_f16
            flash_attn_f32
        )
        set(GGML_OPENCL_KERNEL_DIR ${CPP_DIR}/ggml-opencl/kernels)
        set(GGML_OPENCL_KERNEL_HEADERS "")

        foreach(kernel ${GGML_OPENCL_KERNELS})
            set(input_file ${GGML_OPENCL_KERNEL_DIR}/${kernel}.cl)
            set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${kernel}.cl.h)
            message(STATUS "opencl: embedding kernel ${kernel}")
            add_custom_command(
                OUTPUT ${output_file}
                COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/..
                        ${Python3_EXECUTABLE} ${CPP_DIR}/ggml-opencl/kernels/embed_kernel.py
                        ${input_file} ${output_file}
                DEPENDS ${input_file} ${CPP_DIR}/ggml-opencl/kernels/embed_kernel.py
                COMMENT "Embedding OpenCL kernel: ${kernel}.cl"
            )

            list(APPEND GGML_OPENCL_KERNEL_HEADERS ${output_file})
        endforeach()

        set(OPENCL_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../bin/${ANDROID_ABI})
        set(OPENCL_STUB     ${OPENCL_STUB_DIR}/libOpenCL.so)

        if (EXISTS ${OPENCL_STUB})
            # Keep using -lOpenCL so we don't create a packagable .so target.
            target_link_directories(${jni_name} PRIVATE ${OPENCL_STUB_DIR})
            target_link_libraries(${jni_name} PRIVATE OpenCL)
        else()
            message(WARNING
                "OpenCL library not found. Please build with ./scripts/build-opencl.sh")
        endif()

        target_include_directories(${jni_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        target_sources(${jni_name} PRIVATE
            ${CPP_DIR}/ggml-opencl/ggml-opencl.cpp
            ${GGML_OPENCL_KERNEL_HEADERS}
        )
        target_compile_options(${jni_name} PRIVATE
            -DLM_GGML_USE_OPENCL
            -DLM_GGML_OPENCL_USE_ADRENO_KERNELS
            -DLM_GGML_OPENCL_EMBED_KERNELS
            -DLM_GGML_OPENCL_SOA_Q
        )
    endif ()

    # Hexagon backend support
    # Note: Hexagon requires building DSP libraries with Hexagon SDK toolchain
    # This is a complex process that needs to be done separately
    if (ENABLE_HEXAGON)
        # Hexagon SDK is ALWAYS required to link cdsprpc, even for prebuilt distribution
        set(HEXAGON_SDK_AVAILABLE OFF)
        if (HEXAGON_SDK_ROOT AND HEXAGON_TOOLS_ROOT)
            # Verify libcdsprpc.so exists
            set(CDSPRPC_LIB "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")
            if (EXISTS ${CDSPRPC_LIB})
                set(HEXAGON_SDK_AVAILABLE ON)
                message(STATUS "Hexagon SDK verified and available")
            else()
                message(WARNING
                    "Hexagon SDK found but libcdsprpc.so is missing at ${CDSPRPC_LIB}.\n"
                    "The SDK installation may be incomplete. Run 'npm run bootstrap' to reinstall.")
            endif()
        else()
            message(WARNING
                "Hexagon SDK is required to build ${jni_name} but not found.")
        endif()

        if (HEXAGON_SDK_AVAILABLE)
            # Add host-side hexagon sources
            target_sources(${jni_name} PRIVATE
                ${CPP_DIR}/ggml-hexagon/ggml-hexagon.cpp
                ${CPP_DIR}/ggml-hexagon/htp-drv.cpp
            )

            # Use pre-built stub (always present in source tree)
            target_sources(${jni_name} PRIVATE
                ${CPP_DIR}/ggml-hexagon/htp/v73/htp_iface_stub.c
            )

            # Use pre-built IDL headers from build-hexagon-htp directory
            set(HEXAGON_PREBUILT_HTP_DIR "${CPP_DIR}/ggml-hexagon/htp/v73/")

            target_include_directories(${jni_name} PRIVATE
                ${HEXAGON_SDK_ROOT}/incs
                ${HEXAGON_SDK_ROOT}/incs/stddef
                ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
                ${HEXAGON_SDK_ROOT}/utils/examples
                ${CPP_DIR}/ggml-hexagon
                ${CPP_DIR}/ggml-hexagon/htp
                ${HEXAGON_PREBUILT_HTP_DIR}
                ${CMAKE_CURRENT_BINARY_DIR}
            )

            # Link Hexagon SDK libraries
            # These are stub libraries that communicate with the DSP
            # libcdsprpc.so contains all needed symbols: remote_handle64_*, dspqueue_*, rpcmem_*
            # Note: CDSPRPC_LIB is already verified to exist above
            target_link_libraries(${jni_name} PRIVATE ${CDSPRPC_LIB})
            message(STATUS "Linking Hexagon cdsprpc library: ${CDSPRPC_LIB}")

            target_compile_options(${jni_name} PRIVATE
                -DLM_GGML_USE_HEXAGON
            )

            message(STATUS "Hexagon backend enabled for ${jni_name}")
            if (NOT AJLLAMA_STANDALONE_BUILD)
                message(STATUS "  Note: DSP libraries (libggml-htp-*.so) must be built separately")
                message(STATUS "  Run: scripts/build-hexagon-htp.sh to build DSP libraries")
            endif()
        else()
            # Only show warning when building from source
            if (NOT AJLLAMA_STANDALONE_BUILD)
                message(WARNING
                    "Hexagon SDK not found. Hexagon backend will not be built.")
            endif()
        endif()
    endif ()

endfunction()

# --- Helper function to check if a variant should be built ---
function(should_build_variant variant_name result_var)
    # AJLLAMA_VARIANT can be:
    # - Empty string ("") = build all variants
    # - Specific variant name = build only that variant
    # Valid variant names: "generic", "v8", "v8_2", "v8_2_dotprod", "v8_2_i8mm", 
    #                      "v8_2_dotprod_i8mm", "v8_2_dotprod_i8mm_hexagon_opencl", "x86_64"
    
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
# Default to building from source if not specified
option(AJLLAMA_BUILD_FROM_SOURCE "Build native libraries from source code" ON)

if (AJLLAMA_BUILD_FROM_SOURCE)
    # Always build the generic variant (baseline compatibility)
    should_build_variant("generic" BUILD_GENERIC)
    if (BUILD_GENERIC)
        build_ajllama_jni("ajllama_jni" "generic" "")
        message(STATUS "Building variant: generic")
    endif()

    # ARM64 architecture-specific variants
    if (ANDROID_ABI AND ANDROID_ABI STREQUAL "arm64-v8a")
        should_build_variant("v8" BUILD_V8)
        if (BUILD_V8)
            build_ajllama_jni("ajllama_jni_v8" "arm" "-march=armv8-a")
            message(STATUS "Building variant: v8")
        endif()

        should_build_variant("v8_2" BUILD_V8_2)
        if (BUILD_V8_2)
            build_ajllama_jni("ajllama_jni_v8_2" "arm" "-march=armv8.2-a")
            message(STATUS "Building variant: v8_2")
        endif()

        should_build_variant("v8_2_dotprod" BUILD_V8_2_DOTPROD)
        if (BUILD_V8_2_DOTPROD)
            build_ajllama_jni("ajllama_jni_v8_2_dotprod" "arm" "-march=armv8.2-a+dotprod")
            message(STATUS "Building variant: v8_2_dotprod")
        endif()

        should_build_variant("v8_2_i8mm" BUILD_V8_2_I8MM)
        if (BUILD_V8_2_I8MM)
            build_ajllama_jni("ajllama_jni_v8_2_i8mm" "arm" "-march=armv8.2-a+i8mm")
            message(STATUS "Building variant: v8_2_i8mm")
        endif()

        should_build_variant("v8_2_dotprod_i8mm" BUILD_V8_2_DOTPROD_I8MM)
        if (BUILD_V8_2_DOTPROD_I8MM)
            build_ajllama_jni("ajllama_jni_v8_2_dotprod_i8mm" "arm" "-march=armv8.2-a+dotprod+i8mm")
            message(STATUS "Building variant: v8_2_dotprod_i8mm")
        endif()

        should_build_variant("v8_2_dotprod_i8mm_hexagon_opencl" BUILD_HEXAGON_OPENCL)
        if (BUILD_HEXAGON_OPENCL)
            build_ajllama_jni("ajllama_jni_v8_2_dotprod_i8mm_hexagon_opencl" "arm" "-march=armv8.2-a+dotprod+i8mm")
            message(STATUS "Building variant: v8_2_dotprod_i8mm_hexagon_opencl")
        endif()

    # x86_64 architecture-specific variant
    elseif (ANDROID_ABI AND ANDROID_ABI STREQUAL "x86_64")
        should_build_variant("x86_64" BUILD_X86_64)
        if (BUILD_X86_64)
            build_ajllama_jni("ajllama_jni_x86_64" "x86" "-march=x86-64;-mtune=generic;-msse4.2;-mpopcnt")
            message(STATUS "Building variant: x86_64")
        endif()
    endif()
else()
    message(STATUS "AJLLAMA_BUILD_FROM_SOURCE is OFF. Assuming prebuilt libraries will be used.")
endif()
