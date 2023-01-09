cmake_minimum_required(VERSION 3.17.0)
project(nw-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.2.1
)

FetchContent_MakeAvailable(Catch2)

include(CTest)
include(Catch)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/vendor/cxxopts/include
)

set(NW_COMMON_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/nw/aligner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/nw/creator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/nw/cuda.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/nw/input.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/nw/kernel.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/nw/serial.cpp
)

add_executable(nw-score
    ${CMAKE_CURRENT_LIST_DIR}/examples/score.cpp
    ${NW_COMMON_SOURCES}
)

add_executable(nw-align
    ${CMAKE_CURRENT_LIST_DIR}/examples/align.cpp
    ${NW_COMMON_SOURCES}
)

add_executable(nw-test
    ${CMAKE_CURRENT_LIST_DIR}/tests/cuda.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tests/serial.cpp
    ${NW_COMMON_SOURCES}
)

target_link_libraries(nw-test PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(nw-test)
