# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for workspace and owned tensor leak detection."""

import jax
import jax_tvm_ffi
import numpy
import pytest
import tvm_ffi.cpp
from conftest import requires_gpu
from jax import numpy as jnp


def test_workspace_leak_detection_cpu():
    """Test that retaining workspace tensors beyond FFI call is detected on CPU."""
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="workspace_leak_cpu",
        cpp_sources=r"""
            #include <tvm/ffi/container/tensor.h>
            #include <tvm/ffi/extra/c_env_api.h>

            static tvm::ffi::Tensor leaked_tensor;

            void workspace_leak_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              int64_t n = x.size(0);
              DLDataType dtype = {kDLFloat, 32, 1};
              tvm::ffi::Tensor temp = tvm::ffi::Tensor::FromEnvAlloc(
                  TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
              );
              // INTENTIONAL LEAK: prevents deleter from running
              leaked_tensor = temp;
              for (int i = 0; i < n; ++i) {
                static_cast<float*>(temp.data_ptr())[i] = 2.0f;
                static_cast<float*>(y.data_ptr())[i] =
                    static_cast<float*>(x.data_ptr())[i] +
                    static_cast<float*>(temp.data_ptr())[i];
              }
            }

            void workspace_no_leak_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              int64_t n = x.size(0);
              DLDataType dtype = {kDLFloat, 32, 1};
              tvm::ffi::Tensor temp = tvm::ffi::Tensor::FromEnvAlloc(
                  TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
              );
              for (int i = 0; i < n; ++i) {
                static_cast<float*>(temp.data_ptr())[i] = 2.0f;
                static_cast<float*>(y.data_ptr())[i] =
                    static_cast<float*>(x.data_ptr())[i] +
                    static_cast<float*>(temp.data_ptr())[i];
              }
            }
        """,
        functions=["workspace_leak_cpu", "workspace_no_leak_cpu"],
    )

    jax_tvm_ffi.register_ffi_target(
        "workspace_no_leak_cpu",
        mod.workspace_no_leak_cpu,
        platform="cpu",
        use_last_output_for_alloc_workspace=True,
    )

    x = jnp.arange(10, device=jax.devices("cpu")[0], dtype=jnp.float32)
    workspace_size = x.shape[0] * 4

    results = jax.ffi.ffi_call(
        "workspace_no_leak_cpu",
        (
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct((workspace_size,), jnp.uint8),
        ),
    )(x)

    numpy.testing.assert_allclose(numpy.array(results[0]), numpy.array(x + 2.0))

    jax_tvm_ffi.register_ffi_target(
        "workspace_leak_cpu",
        mod.workspace_leak_cpu,
        platform="cpu",
        use_last_output_for_alloc_workspace=True,
    )

    with pytest.raises(Exception, match="(?i)workspace|leak"):
        jax.ffi.ffi_call(
            "workspace_leak_cpu",
            (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((workspace_size,), jnp.uint8),
            ),
        )(x)


@requires_gpu
def test_workspace_leak_detection_gpu():
    """Test that retaining workspace tensors beyond FFI call is detected on GPU."""
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="workspace_leak_gpu",
        cuda_sources=r"""
        #include <cuda_runtime.h>
        #include <tvm/ffi/container/tensor.h>
        #include <tvm/ffi/extra/c_env_api.h>

        static tvm::ffi::Tensor leaked_gpu_tensor;

        __global__ void add_kernel(const float* x, const float* temp, float* y, int n) {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if (idx < n) {
            y[idx] = x[idx] + temp[idx];
          }
        }

        void workspace_leak_gpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
          int64_t n = x.size(0);
          DLDataType dtype = {kDLFloat, 32, 1};
          tvm::ffi::Tensor temp = tvm::ffi::Tensor::FromEnvAlloc(
              TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
          );
          // INTENTIONAL LEAK
          leaked_gpu_tensor = temp;
          cudaMemset(temp.data_ptr(), 0, n * sizeof(float));
          float val = 2.0f;
          for (int i = 0; i < n; ++i) {
            cudaMemcpy(static_cast<float*>(temp.data_ptr()) + i, &val,
                       sizeof(float), cudaMemcpyHostToDevice);
          }
          int threads = 256;
          int blocks = (n + threads - 1) / threads;
          add_kernel<<<blocks, threads>>>(
            static_cast<const float*>(x.data_ptr()),
            static_cast<const float*>(temp.data_ptr()),
            static_cast<float*>(y.data_ptr()),
            n
          );
        }

        void workspace_no_leak_gpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
          int64_t n = x.size(0);
          DLDataType dtype = {kDLFloat, 32, 1};
          tvm::ffi::Tensor temp = tvm::ffi::Tensor::FromEnvAlloc(
              TVMFFIEnvTensorAlloc, {n}, dtype, x.device()
          );
          cudaMemset(temp.data_ptr(), 0, n * sizeof(float));
          float val = 2.0f;
          for (int i = 0; i < n; ++i) {
            cudaMemcpy(static_cast<float*>(temp.data_ptr()) + i, &val,
                       sizeof(float), cudaMemcpyHostToDevice);
          }
          int threads = 256;
          int blocks = (n + threads - 1) / threads;
          add_kernel<<<blocks, threads>>>(
            static_cast<const float*>(x.data_ptr()),
            static_cast<const float*>(temp.data_ptr()),
            static_cast<float*>(y.data_ptr()),
            n
          );
        }
    """,
        functions=["workspace_leak_gpu", "workspace_no_leak_gpu"],
    )

    jax_tvm_ffi.register_ffi_target(
        "workspace_no_leak_gpu",
        mod.workspace_no_leak_gpu,
        platform="gpu",
        use_last_output_for_alloc_workspace=True,
    )

    x = jnp.arange(256, device=jax.devices("gpu")[0], dtype=jnp.float32)
    workspace_size = x.shape[0] * 4

    results = jax.ffi.ffi_call(
        "workspace_no_leak_gpu",
        (
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct((workspace_size,), jnp.uint8),
        ),
    )(x)

    numpy.testing.assert_allclose(numpy.array(results[0]), numpy.array(x + 2.0), rtol=1e-5)

    jax_tvm_ffi.register_ffi_target(
        "workspace_leak_gpu",
        mod.workspace_leak_gpu,
        platform="gpu",
        use_last_output_for_alloc_workspace=True,
    )

    with pytest.raises(Exception, match="(?i)workspace|leak"):
        results = jax.ffi.ffi_call(
            "workspace_leak_gpu",
            (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((workspace_size,), jnp.uint8),
            ),
        )(x)
        jax.block_until_ready(results)


def test_owned_tensor_leak_detection():
    """Test that retaining owned tensors beyond FFI call is detected."""
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="owned_tensor_leak",
        cpp_sources=r"""
            #include <tvm/ffi/container/tensor.h>

            static tvm::ffi::Tensor leaked_owned_tensor;

            void owned_tensor_leak(tvm::ffi::Tensor x, tvm::ffi::Tensor y) {
              int64_t n = x.size(0);
              // INTENTIONAL LEAK
              leaked_owned_tensor = x;
              for (int i = 0; i < n; ++i) {
                static_cast<float*>(y.data_ptr())[i] =
                    static_cast<float*>(x.data_ptr())[i] + 1.0f;
              }
            }

            void owned_tensor_no_leak(tvm::ffi::Tensor x, tvm::ffi::Tensor y) {
              int64_t n = x.size(0);
              for (int i = 0; i < n; ++i) {
                static_cast<float*>(y.data_ptr())[i] =
                    static_cast<float*>(x.data_ptr())[i] + 1.0f;
              }
            }
        """,
        functions=["owned_tensor_leak", "owned_tensor_no_leak"],
    )

    jax_tvm_ffi.register_ffi_target(
        "owned_tensor_no_leak",
        mod.owned_tensor_no_leak,
        platform="cpu",
        pass_owned_tensor=True,
    )

    x = jnp.arange(10, device=jax.devices("cpu")[0], dtype=jnp.float32)

    result = jax.ffi.ffi_call(
        "owned_tensor_no_leak",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(x)

    numpy.testing.assert_allclose(numpy.array(result), numpy.array(x + 1.0))

    jax_tvm_ffi.register_ffi_target(
        "owned_tensor_leak",
        mod.owned_tensor_leak,
        platform="cpu",
        pass_owned_tensor=True,
    )

    with pytest.raises(Exception, match="(?i)leak|retain"):
        jax.ffi.ffi_call(
            "owned_tensor_leak",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x)
