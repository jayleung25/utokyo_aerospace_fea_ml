cmake_minimum_required(VERSION 3.16)
project(czm_preprocessor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Static library (link into FEA solver or inference wrapper) ----
add_library(czm_preprocessor STATIC
    preprocessor.cpp
)
target_include_directories(czm_preprocessor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Standalone test executable ----
add_executable(preprocessor_test
    preprocessor.cpp
)
target_include_directories(preprocessor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(preprocessor_test PRIVATE CZM_PREPROCESSOR_MAIN)

# Compiler warnings
if(MSVC)
    target_compile_options(czm_preprocessor PRIVATE /W4)
    target_compile_options(preprocessor_test PRIVATE /W4)
else()
    target_compile_options(czm_preprocessor PRIVATE -Wall -Wextra -O2)
    target_compile_options(preprocessor_test PRIVATE -Wall -Wextra -O2)
endif()
