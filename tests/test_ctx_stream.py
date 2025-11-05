# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ctx.stream support in JAX-TVM-FFI"""

import jax
import jax.numpy as jnp
import jax_tvm_ffi
import numpy
import tvm_ffi
import tvm_ffi.cpp
from conftest import requires_gpu


@requires_gpu
def test_ctx_stream_gpu_python_callback():
    """Test that ctx.stream passes non-zero stream pointer on GPU with Python callback"""
    call_counter = [0]
    received_stream = [None]

    def check_stream_gpu(x, y, stream):
        """Callback that receives stream as int64_t"""
        call_counter[0] += 1
        received_stream[0] = stream
        # On GPU, stream should be non-zero pointer
        assert isinstance(stream, int), f"Stream should be int, got {type(stream)}"
        assert stream != 0, "GPU stream should be non-zero pointer"
        # Just verify we received the tensors
        assert isinstance(x, tvm_ffi.Tensor)
        assert isinstance(y, tvm_ffi.Tensor)

    jax_tvm_ffi.register_ffi_target(
        "testing.check_stream_gpu",
        check_stream_gpu,
        ["args", "rets", "ctx.stream"],
        platform="gpu",
        pass_owned_tensor=True,
    )

    x = jnp.arange(10, device=jax.devices("gpu")[0], dtype=jnp.float32)
    _ = jax.ffi.ffi_call("testing.check_stream_gpu", jax.ShapeDtypeStruct(x.shape, x.dtype))(x)

    assert call_counter[0] == 1, "Function should be called once"
    assert received_stream[0] != 0, f"GPU stream should be non-zero, got {received_stream[0]}"


@requires_gpu
def test_ctx_stream_with_attrs():
    """Test that ctx.stream works together with attributes on GPU"""
    call_counter = [0]
    received_values = {}

    def check_stream_and_attrs(x, y, threshold, mode, stream):
        """Callback that receives both attributes and stream"""
        call_counter[0] += 1
        received_values["stream"] = stream
        received_values["threshold"] = threshold
        received_values["mode"] = mode

        assert isinstance(stream, int), f"Stream should be int, got {type(stream)}"
        assert stream != 0, f"GPU stream should be non-zero, got {stream}"
        assert threshold == 0.5, f"Expected threshold=0.5, got {threshold}"
        assert mode == "fast", f"Expected mode='fast', got {mode}"

        # Just verify we received the tensors
        assert isinstance(x, tvm_ffi.Tensor)
        assert isinstance(y, tvm_ffi.Tensor)

    jax_tvm_ffi.register_ffi_target(
        "testing.check_stream_and_attrs",
        check_stream_and_attrs,
        ["args", "rets", "attrs.threshold", "attrs.mode", "ctx.stream"],
        platform="gpu",
        pass_owned_tensor=True,
    )

    x = jnp.arange(10, device=jax.devices("gpu")[0], dtype=jnp.float32)
    _ = jax.ffi.ffi_call("testing.check_stream_and_attrs", jax.ShapeDtypeStruct(x.shape, x.dtype))(
        x, threshold=0.5, mode="fast"
    )

    assert call_counter[0] == 1, "Function should be called once"
    assert received_values["stream"] != 0, (
        f"GPU stream should be non-zero, got {received_values['stream']}"
    )
    assert received_values["threshold"] == 0.5, "Threshold should be 0.5"
    assert received_values["mode"] == "fast", "Mode should be 'fast'"


@requires_gpu
def test_ctx_stream_inline_cuda_kernel():
    """Test ctx.stream with inline compiled CUDA kernel on GPU"""
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="stream_test_cuda",
        cuda_sources=r"""
            #include <cuda_runtime.h>

            __global__ void add_one_kernel(const float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1.0f;
              }
            }

            void kernel_with_stream(tvm::ffi::TensorView x, tvm::ffi::TensorView y, int64_t stream) {
              // Verify we receive stream as int64_t
              printf("Received GPU stream pointer: %lld (0x%llx)\n", (long long)stream, (unsigned long long)stream);

              TVM_FFI_ICHECK(stream != 0) << "GPU stream should be non-zero";
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";

              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";

              // Launch CUDA kernel using the provided stream
              int n = x.size(0);
              int threads = 256;
              int blocks = (n + threads - 1) / threads;

              cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
              add_one_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(x.data_ptr()),
                static_cast<float*>(y.data_ptr()),
                n
              );
            }
        """,
        functions=["kernel_with_stream"],
    )

    jax_tvm_ffi.register_ffi_target(
        "kernel_with_stream_cuda",
        mod.kernel_with_stream,
        ["args", "rets", "ctx.stream"],
        platform="gpu",
    )

    x = jnp.arange(10, device=jax.devices("gpu")[0], dtype=jnp.float32)

    @jax.jit
    def call_kernel_with_stream(x):
        return jax.ffi.ffi_call(
            "kernel_with_stream_cuda",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    y = call_kernel_with_stream(x)
    numpy.testing.assert_allclose(jnp.array(x + 1), jnp.array(y), rtol=1e-5)
