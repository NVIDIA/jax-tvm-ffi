# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(jax_tvm_ffi)

option(JAX_TVM_FFI_SHIP_DEBUG_SYMBOLS "Ship debug symbols" ON)

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
find_package(tvm_ffi CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE XLA_DIR
)
message(STATUS "XLA include directory: ${XLA_DIR}")

# use the projects as usual
add_library(jax_tvm_ffi SHARED src/jax_tvm_ffi.cc)
target_include_directories(jax_tvm_ffi PRIVATE ${XLA_DIR})
target_link_libraries(jax_tvm_ffi tvm_ffi::header)
target_link_libraries(jax_tvm_ffi tvm_ffi::shared)

# show as jax_tvm_ffi.so
set_target_properties(jax_tvm_ffi PROPERTIES PREFIX "")

if (JAX_TVM_FFI_SHIP_DEBUG_SYMBOLS)
  # ship debugging symbols for backtrace on macos
  tvm_ffi_add_prefix_map(jax_tvm_ffi ${CMAKE_CURRENT_SOURCE_DIR})
  tvm_ffi_add_apple_dsymutil(jax_tvm_ffi)
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
    DESTINATION .
    FILES_MATCHING
    PATTERN "*.dSYM"
  )
endif ()

install(TARGETS jax_tvm_ffi DESTINATION .)
