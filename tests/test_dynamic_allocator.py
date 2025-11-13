# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for XLA scratch allocator (dynamic allocation without pre-allocated workspace)."""

import jax
import jax_tvm_ffi
import numpy
import tvm_ffi.cpp
from conftest import requires_gpu
from jax import numpy as jnp

# NOTE: XLA scratch allocator only works on GPU, not CPU
# CPU doesn't have a "device memory allocator" in XLA FFI terms
# For CPU, must use static workspace with use_last_output_for_alloc_workspace=True


@requires_gpu
def test_scratch_allocator_gpu():
    """Test XLA scratch allocator on GPU (no workspace required, no CUDA graph)"""
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="scratch_gpu_add",
        cuda_sources=r"""
        #include <cuda_runtime.h>

        __global__ void add_kernel(const float* x, float* temp, float* y, int n) {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if (idx < n) {
            temp[idx] = x[idx] * 2.0f;
            y[idx] = temp[idx] + 1.0f;
          }
        }

        void scratch_gpu_add(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
          int64_t n = x.size(0);
          DLDataType dtype = {kDLFloat, 32, 1};

          // Allocate temp buffer using XLA scratch allocator
          tvm::ffi::Tensor temp = tvm::ffi::Tensor::FromEnvAlloc(
              TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
          );

          int threads = 256;
          int blocks = (n + threads - 1) / threads;
          add_kernel<<<blocks, threads>>>(
            static_cast<const float*>(x.data_ptr()),
            static_cast<float*>(temp.data_ptr()),
            static_cast<float*>(y.data_ptr()),
            n
          );
        }
    """,
        functions=["scratch_gpu_add"],
    )

    # Register WITHOUT workspace and WITHOUT allow_cuda_graph
    jax_tvm_ffi.register_ffi_target(
        "scratch_gpu_add",
        mod.scratch_gpu_add,
        platform="gpu",
        allow_cuda_graph=False,  # Can't use CUDA graph with dynamic allocation
        use_last_output_for_alloc_workspace=False,
    )

    x = jnp.arange(1024, device=jax.devices("gpu")[0], dtype=jnp.float32)

    # Call without workspace
    result = jax.ffi.ffi_call(
        "scratch_gpu_add",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(x)

    # Expected: y = 2x + 1
    expected = numpy.array(x) * 2.0 + 1.0
    numpy.testing.assert_allclose(numpy.array(result), expected)

    # Peak should be 0 for scratch allocator
    peak_usage = jax_tvm_ffi.get_last_workspace_peak()
    assert peak_usage == 0, f"Expected 0 for scratch allocator, got {peak_usage}"

    @jax.jit
    def jit_compute(x):
        return jax.ffi.ffi_call("scratch_gpu_add", jax.ShapeDtypeStruct(x.shape, x.dtype))(x)

    result_jit = jit_compute(x)
    numpy.testing.assert_allclose(numpy.array(result_jit), expected)


@requires_gpu
def test_scratch_allocator_multiple_allocations():
    """Test XLA scratch allocator with multiple allocations in one kernel"""
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="scratch_multi_alloc",
        cuda_sources=r"""
        #include <cuda_runtime.h>

        __global__ void multi_kernel(const float* x, float* temp1, float* temp2, float* y, int n) {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if (idx < n) {
            temp1[idx] = x[idx] * 2.0f;
            temp2[idx] = x[idx] + 5.0f;
            y[idx] = temp1[idx] + temp2[idx];
          }
        }

        void scratch_multi_alloc(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
          int64_t n = x.size(0);
          DLDataType dtype = {kDLFloat, 32, 1};

          // Allocate TWO temp buffers using XLA scratch allocator
          tvm::ffi::Tensor temp1 = tvm::ffi::Tensor::FromEnvAlloc(
              TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
          );

          tvm::ffi::Tensor temp2 = tvm::ffi::Tensor::FromEnvAlloc(
              TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
          );

          int threads = 256;
          int blocks = (n + threads - 1) / threads;
          multi_kernel<<<blocks, threads>>>(
            static_cast<const float*>(x.data_ptr()),
            static_cast<float*>(temp1.data_ptr()),
            static_cast<float*>(temp2.data_ptr()),
            static_cast<float*>(y.data_ptr()),
            n
          );
        }
    """,
        functions=["scratch_multi_alloc"],
    )

    jax_tvm_ffi.register_ffi_target(
        "scratch_multi_alloc",
        mod.scratch_multi_alloc,
        platform="gpu",
        allow_cuda_graph=False,
        use_last_output_for_alloc_workspace=False,
    )

    x = jnp.arange(512, device=jax.devices("gpu")[0], dtype=jnp.float32)

    result = jax.ffi.ffi_call(
        "scratch_multi_alloc",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(x)

    # Expected: y = 2x + (x + 5) = 3x + 5
    expected = numpy.array(x) * 3.0 + 5.0
    numpy.testing.assert_allclose(numpy.array(result), expected)
