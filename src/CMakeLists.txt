set(HAVE_GCS 0 CACHE BOOL "Have Google Cloud Storage SDK")

include_directories(commons)
add_subdirectory(strucclustutils)
add_subdirectory(commons)
add_subdirectory(workflow)

add_library(foldseek-framework
        ${commons_source_files}
        ${strucclustutils_source_files}
        ${workflow_source_files}
        LocalCommandDeclarations.h
        FoldseekBase.cpp
)
mmseqs_setup_derived_target(foldseek-framework)
target_link_libraries(foldseek-framework gemmiwrapper 3di pulchra kerasify tmalign)
if(ENABLE_STRUCTTY)
        target_link_libraries(foldseek-framework structty)
        target_compile_definitions(foldseek-framework PUBLIC -DHAVE_STRUCTTY=1)
endif()
if(ENABLE_MODERNPROST)
        add_library(modernprost strucclustutils/ModernProst.h strucclustutils/ModernProst.cpp)
        target_include_directories(modernprost PUBLIC ${CMAKE_SOURCE_DIR}/lib/ggml/include)
        target_link_libraries(modernprost ggml ggml-base ggml-cpu)
        if(ENABLE_CUDA)
                target_compile_definitions(modernprost PRIVATE -DGGML_USE_CUDA=1)
                target_link_libraries(modernprost ggml-cuda)
        endif()
        target_link_libraries(foldseek-framework modernprost)
        target_compile_definitions(foldseek-framework PUBLIC -DHAVE_MODERNPROST=1)
endif()
if(ENABLE_PROSTT5)
        add_library(prostt5 ${prostt5_source_files})
        target_include_directories(prostt5 PUBLIC ${CMAKE_SOURCE_DIR}/lib/ggml/include)
        target_link_libraries(prostt5 ggml ggml-base ggml-cpu)
        if(ENABLE_CUDA)
                target_link_libraries(prostt5 ggml-cuda)
        endif()
        target_link_libraries(foldseek-framework prostt5)
        target_compile_definitions(foldseek-framework PUBLIC -DHAVE_PROSTT5=1)
endif()

if(ENABLE_CUDA)
        target_compile_definitions(foldseek-framework PUBLIC -DHAVE_CUDA=1)
endif()


add_dependencies(foldseek-framework local-generated)

if(HAVE_GCS)
        find_package(google_cloud_cpp_storage REQUIRED)
        target_link_libraries(foldseek-framework google-cloud-cpp::storage)
        target_compile_definitions(foldseek-framework PUBLIC HAVE_GCS)
        # google-cloud-cpp requires C++17
        set_property(SOURCE strucclustutils/structcreatedb.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -std=c++17")
endif()

if (NOT FOLDSEEK_FRAMEWORK_ONLY)
        add_subdirectory(version)
        add_executable(foldseek foldseek.cpp)
        mmseqs_setup_derived_target(foldseek foldseek-framework)
        target_link_libraries(mmseqs-framework version)
        target_link_libraries(foldseek version)
        install(TARGETS foldseek DESTINATION bin)
endif()
