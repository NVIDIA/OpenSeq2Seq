cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(External-LM)

find_package(Torch REQUIRED)

add_executable(extlm extlm.cpp)
target_link_libraries(extlm "${TORCH_LIBRARIES}")
set_property(TARGET extlm PROPERTY CXX_STANDARD 11)
