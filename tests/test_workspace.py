# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import jax
import jax_tvm_ffi
import numpy
import tvm_ffi.cpp
from conftest import requires_gpu
from jax import numpy as jnp


def test_workspace_cpu():
    """Test workspace allocation on CPU with verification that workspace is actually used"""
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="workspace_add",
        cpp_sources=r"""
            void workspace_add(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              int64_t n = x.size(0);
              DLDataType dtype = {kDLFloat, 32, 1};
              // Allocate temp tensor from workspace
              tvm::ffi::Tensor temp = tvm::ffi::Tensor::FromEnvAlloc(
                  TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
              );
              // Use the temp tensor
              for (int i = 0; i < n; ++i) {
                static_cast<float*>(temp.data_ptr())[i] = 2.0f;
                static_cast<float*>(y.data_ptr())[i] =
                    static_cast<float*>(x.data_ptr())[i] +
                    static_cast<float*>(temp.data_ptr())[i];
              }
            }
        """,
        functions=["workspace_add"],
    )

    jax_tvm_ffi.register_ffi_target(
        "workspace_add", mod.workspace_add, platform="cpu", use_last_output_for_alloc_workspace=True
    )

    x = jnp.arange(10, device=jax.devices("cpu")[0], dtype=jnp.float32)
    workspace_size = x.shape[0] * 4  # 10 floats * 4 bytes = 40 bytes

    # Call with explicit workspace in output tuple
    results = jax.ffi.ffi_call(
        "workspace_add",
        (
            jax.ShapeDtypeStruct(x.shape, x.dtype),  # Actual output
            jax.ShapeDtypeStruct((workspace_size,), jnp.uint8),  # Workspace
        ),
    )(x)

    # Verify result
    result = results[0]
    numpy.testing.assert_allclose(numpy.array(result), numpy.array(x + 2.0))

    # Verify workspace was actually used
    peak_usage = jax_tvm_ffi.get_last_workspace_peak()
    assert peak_usage == 40, f"Expected 40 bytes workspace usage, got {peak_usage}"

    # Test with JIT to ensure no side effects
    @jax.jit
    def jit_compute(x):
        results = jax.ffi.ffi_call(
            "workspace_add",
            (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((workspace_size,), jnp.uint8),
            ),
        )(x)
        return results[0]

    result_jit = jit_compute(x)
    numpy.testing.assert_allclose(numpy.array(result_jit), numpy.array(x + 2.0))


@requires_gpu
def test_workspace_gpu():
    """Test workspace allocation on GPU with CUDA graphs (enabled by default for FFI)"""
    # CUDA kernel with multiple workspace allocations for complex computation
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="workspace_gpu_compute",
        cuda_sources=r"""
        #include <cuda_runtime.h>

        // Kernel that uses two temp buffers - harder to optimize away
        __global__ void compute_kernel(const float* x, float* temp1, float* temp2,
                                       float* y, int n) {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if (idx < n) {
            // Stage 1: Write to temp1 (x * 3)
            temp1[idx] = x[idx] * 3.0f;

            // Stage 2: Write to temp2 (x + 1)
            temp2[idx] = x[idx] + 1.0f;

            // Stage 3: Combine both temps (can't be optimized away!)
            // Forces both buffers to be materialized and read back
            y[idx] = temp1[idx] + temp2[idx];
          }
        }

        void workspace_gpu_compute(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
          int64_t n = x.size(0);
          DLDataType dtype = {kDLFloat, 32, 1};

          // Allocate TWO temp buffers from workspace (not cudaMalloc!)
          // This proves multiple allocations work and are tracked
          tvm::ffi::Tensor temp1 = tvm::ffi::Tensor::FromEnvAlloc(
              TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
          );

          tvm::ffi::Tensor temp2 = tvm::ffi::Tensor::FromEnvAlloc(
              TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
          );

          // Launch CUDA kernel using both workspace buffers
          int threads = 256;
          int blocks = (n + threads - 1) / threads;
          compute_kernel<<<blocks, threads>>>(
            static_cast<const float*>(x.data_ptr()),
            static_cast<float*>(temp1.data_ptr()),
            static_cast<float*>(temp2.data_ptr()),
            static_cast<float*>(y.data_ptr()),
            n
          );
        }
    """,
        functions=["workspace_gpu_compute"],
    )

    jax_tvm_ffi.register_ffi_target(
        "workspace_gpu_compute",
        mod.workspace_gpu_compute,
        platform="gpu",
        allow_cuda_graph=True,
        use_last_output_for_alloc_workspace=True,
    )

    x = jnp.arange(1024, device=jax.devices("gpu")[0], dtype=jnp.float32)
    # Need workspace for TWO temp buffers: 1024 floats * 4 bytes * 2 = 8192 bytes
    workspace_size = x.shape[0] * 4 * 2

    # Direct call
    results = jax.ffi.ffi_call(
        "workspace_gpu_compute",
        (
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct((workspace_size,), jnp.uint8),
        ),
    )(x)

    result = results[0]
    # Expected: y = 3x + (x + 1) = 4x + 1
    expected = numpy.array(x) * 4.0 + 1.0
    numpy.testing.assert_allclose(numpy.array(result), expected)

    peak_usage = jax_tvm_ffi.get_last_workspace_peak()
    assert peak_usage == 8192, f"Expected 8192 bytes (2 buffers), got {peak_usage}"

    @jax.jit
    def jit_compute(x):
        results = jax.ffi.ffi_call(
            "workspace_gpu_compute",
            (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((workspace_size,), jnp.uint8),
            ),
        )(x)
        return results[0]

    _ = jit_compute(x)

    result_jit = jit_compute(x)
    numpy.testing.assert_allclose(numpy.array(result_jit), expected)
