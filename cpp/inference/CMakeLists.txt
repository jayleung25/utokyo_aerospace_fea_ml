cmake_minimum_required(VERSION 3.16)
project(czm_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Optional: path to TensorFlow C library headers and shared lib ----
# Set this when a SavedModel has been exported and TF is installed:
#   cmake -DCZM_TF_AVAILABLE=ON -DTF_LIB_DIR=/path/to/tensorflow/lib ..
option(CZM_TF_AVAILABLE "Link against TensorFlow C API" OFF)

# Pull in the preprocessor (needed by callers who use both together)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../preprocessor
    ${CMAKE_CURRENT_BINARY_DIR}/preprocessor
)

# ---- Shared library — loaded by Abaqus at runtime ----
add_library(czm_surrogate SHARED
    czm_surrogate.cpp
)
target_include_directories(czm_surrogate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(czm_surrogate PRIVATE czm_preprocessor)

if(CZM_TF_AVAILABLE)
    if(NOT DEFINED TF_LIB_DIR)
        message(FATAL_ERROR "CZM_TF_AVAILABLE=ON requires -DTF_LIB_DIR=<path>")
    endif()
    target_compile_definitions(czm_surrogate PRIVATE CZM_TF_AVAILABLE)
    target_include_directories(czm_surrogate PRIVATE ${TF_LIB_DIR}/include)
    target_link_directories(czm_surrogate PRIVATE ${TF_LIB_DIR}/lib)
    target_link_libraries(czm_surrogate PRIVATE tensorflow)
    message(STATUS "CZM: TensorFlow C API enabled from ${TF_LIB_DIR}")
else()
    message(STATUS "CZM: Stub build — TF not linked (set CZM_TF_AVAILABLE=ON after training)")
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(czm_surrogate PRIVATE /W4)
else()
    target_compile_options(czm_surrogate PRIVATE -Wall -Wextra -O2)
endif()
