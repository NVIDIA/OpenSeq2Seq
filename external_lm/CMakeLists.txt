cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(External-LM)

find_package(Torch REQUIRED)

add_library(externallm STATIC external_lm.cpp)
add_executable(extlm extlm.cpp)

target_link_libraries(externallm "${TORCH_LIBRARIES}")
target_link_libraries(extlm externallm)
set_property(TARGET extlm PROPERTY CXX_STANDARD 11)
