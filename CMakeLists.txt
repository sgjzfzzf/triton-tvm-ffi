cmake_minimum_required(VERSION 3.18)

project(${SKBUILD_PROJECT_NAME})

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
    DESTINATION ${CMAKE_INSTALL_PREFIX}/triton_tvm_ffi
)
