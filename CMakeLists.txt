cmake_minimum_required(VERSION 3.20)
project(llama_cpp_mini CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

include_directories(include)

add_executable(llama_mini
    src/main.cpp
    src/gguf.cpp
    src/dequant.cpp
    src/model.cpp
    src/forward.cpp
)
