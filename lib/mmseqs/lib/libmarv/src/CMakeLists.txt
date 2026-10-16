cmake_minimum_required(VERSION 3.24)

if(ENABLE_HIP)
    set(MARV_GPU_LANGUAGE HIP)
else()
    set(MARV_GPU_LANGUAGE CUDA)
endif()

project(marv LANGUAGES CXX ${MARV_GPU_LANGUAGE})

if(ENABLE_HIP)
    find_package(HIP REQUIRED)
else()
    find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIBRARY_ONLY 0 CACHE BOOL "Do not build executables")
set(BUILD_TUNING 0 CACHE BOOL "Build the gapless tuning benchmark tool")
set(MARV_TUNING_GENCODE "" CACHE STRING "Explicit -gencode for the tuner (CUDA only); empty uses CMAKE_CUDA_ARCHITECTURES")

set(NVCC_FLAGS)
list(APPEND NVCC_FLAGS --extended-lambda)
list(APPEND NVCC_FLAGS --expt-relaxed-constexpr)
string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
if(BUILD_TYPE_LOWER STREQUAL "debug" OR BUILD_TYPE_LOWER STREQUAL "relwithdebinfo")
    list(APPEND NVCC_FLAGS -lineinfo)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/MarvGpuArch.cmake)

set(MARV_CORE_GPU_SOURCES
    aligners/make_aligner.cu
    aligners/make_aligner_endpos.cu
)

set(MARV_HIP_ALIGNER_SOURCES
    aligners/make_aligner_defaultptx.cu
    aligners/make_aligner_endpos_defaultptx.cu
    aligners/make_aligner_amd.cu
    aligners/make_aligner_endpos_amd.cu
)

function(configure_marv_gpu_target target)
    target_compile_definitions(${target} PRIVATE NO_NVTOOLSEXT)

    if(ENABLE_HIP)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:-Wno-assume>
            $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-result>
            $<$<COMPILE_LANGUAGE:HIP>:-Wno-uninitialized>
            $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value>
        )
    else()
        target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
        )
    endif()
endfunction()

set(MARV_ALIGNER_OBJECTS "")
set(MARV_HAVE_ARCH_DEFINES "")
set(MARV_CORE_ARCHS "")
set(MARV_HIP_ARCHS "")
if(ENABLE_HIP)
    marv_resolve_hip_architectures(MARV_HIP_ARCHS)
    message(STATUS "marv: HIP architectures ${MARV_HIP_ARCHS}")
else()
    marv_resolve_cuda_architectures(MARV_CUDA_ARCHS)
    marv_core_architectures("${MARV_CUDA_ARCHS}" MARV_CORE_ARCHS)
    message(STATUS "marv: tuned kernels for ${MARV_CUDA_ARCHS}; core architectures ${MARV_CORE_ARCHS}")

    foreach(arch IN LISTS MARV_CUDA_ARCHS ITEMS defaultptx)
        if(arch STREQUAL "defaultptx")
            set(suffix "defaultptx")
            set(gencode_list "-gencode=arch=compute_80,code=compute_80")
        else()
            set(suffix "sm${arch}")
            marv_arch_object_gencode("${arch}" gencode_list)
            list(APPEND MARV_HAVE_ARCH_DEFINES "MARV_HAVE_SM${arch}")
        endif()

        add_library(marv_aligner_${suffix} OBJECT
            aligners/make_aligner_${suffix}.cu
            aligners/make_aligner_endpos_${suffix}.cu
        )
        set_target_properties(marv_aligner_${suffix} PROPERTIES CUDA_ARCHITECTURES OFF)
        target_include_directories(marv_aligner_${suffix} PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_compile_definitions(marv_aligner_${suffix} PRIVATE NO_NVTOOLSEXT)
        target_compile_options(marv_aligner_${suffix} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS};${gencode_list}>
        )
        list(APPEND MARV_ALIGNER_OBJECTS $<TARGET_OBJECTS:marv_aligner_${suffix}>)
    endforeach()
endif()

set(MARV_GPU_SOURCES ${MARV_CORE_GPU_SOURCES})
if(ENABLE_HIP)
    list(APPEND MARV_GPU_SOURCES ${MARV_HIP_ALIGNER_SOURCES})
endif()

set_source_files_properties(${MARV_GPU_SOURCES} marv.cu dbdata.cpp PROPERTIES LANGUAGE ${MARV_GPU_LANGUAGE})

set(MARV_LIB_SOURCES ${MARV_GPU_SOURCES} marv.cu dbdata.cpp marv.h ${MARV_ALIGNER_OBJECTS})

add_library(marv ${MARV_LIB_SOURCES})
configure_marv_gpu_target(marv)
target_compile_definitions(marv PRIVATE ${MARV_HAVE_ARCH_DEFINES})

if(NOT ENABLE_HIP)
    set_target_properties(marv PROPERTIES CUDA_ARCHITECTURES "${MARV_CORE_ARCHS}" CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
    set_target_properties(marv PROPERTIES HIP_ARCHITECTURES "${MARV_HIP_ARCHS}")
endif()

if(NOT LIBRARY_ONLY)
    find_package(ZLIB REQUIRED)
    find_package(OpenMP REQUIRED)

    set(MARV_ALIGN_GPU_SOURCES ${MARV_GPU_SOURCES} main2.cu)
    set_source_files_properties(main2.cu options.cpp makedb.cpp PROPERTIES LANGUAGE ${MARV_GPU_LANGUAGE})

    add_executable(align ${MARV_ALIGN_GPU_SOURCES} dbdata.cpp options.cpp sequence_io.cpp ${MARV_ALIGNER_OBJECTS})
    configure_marv_gpu_target(align)
    target_compile_definitions(align PRIVATE ${MARV_HAVE_ARCH_DEFINES})
    if(NOT ENABLE_HIP)
        set_target_properties(align PROPERTIES CUDA_ARCHITECTURES "${MARV_CORE_ARCHS}" CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        target_compile_options(align PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    else()
        set_target_properties(align PROPERTIES HIP_ARCHITECTURES "${MARV_HIP_ARCHS}")
    endif()
    target_link_libraries(align PRIVATE ZLIB::ZLIB OpenMP::OpenMP_CXX)
    if(NOT ENABLE_HIP)
        if(TARGET CUDA::nvToolsExt)
            target_link_libraries(align PRIVATE CUDA::nvToolsExt)
        endif()
    endif()

    add_executable(makedb makedb.cpp sequence_io.cpp dbdata.cpp)
    configure_marv_gpu_target(makedb)
    target_link_libraries(makedb PRIVATE ZLIB::ZLIB OpenMP::OpenMP_CXX)
endif()

if(BUILD_TUNING)
    find_package(OpenMP REQUIRED)

    set(MARV_TUNING_SOURCES
        tuning/tuning_gapless.cu
        tuning/benchmarkdata.cu
        tuning/benchmark_uint8x4_scoreonly_singleTile.cu
        tuning/benchmark_uint8x4_scoreonly_multiTile.cu
        tuning/benchmark_short2_scoreonly_singleTile.cu
        tuning/benchmark_short2_scoreonly_multiTile.cu
        tuning/benchmark_half2_scoreonly_singleTile.cu
        tuning/benchmark_half2_scoreonly_multiTile.cu
        tuning/benchmark_float_scoreonly_singleTile.cu
        tuning/benchmark_float_scoreonly_multiTile.cu
    )
    set_source_files_properties(${MARV_TUNING_SOURCES} PROPERTIES LANGUAGE ${MARV_GPU_LANGUAGE})

    add_executable(tuning_gapless ${MARV_TUNING_SOURCES})
    configure_marv_gpu_target(tuning_gapless)
    target_include_directories(tuning_gapless PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(tuning_gapless PRIVATE OpenMP::OpenMP_CXX)
    if(NOT ENABLE_HIP)
        set_target_properties(tuning_gapless PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        target_compile_options(tuning_gapless PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
        if(MARV_TUNING_GENCODE)
            string(REPLACE " " ";" MARV_TUNING_GENCODE_LIST "${MARV_TUNING_GENCODE}")
            set_target_properties(tuning_gapless PROPERTIES CUDA_ARCHITECTURES OFF)
            target_compile_options(tuning_gapless PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MARV_TUNING_GENCODE_LIST}>)
        endif()
    endif()
endif()
