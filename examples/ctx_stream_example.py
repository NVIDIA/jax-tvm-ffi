#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example demonstrating ctx.stream support in jax-tvm-ffi.

This shows how to write a custom CUDA kernel that uses the stream passed from JAX.
"""

import jax
import jax.numpy as jnp
import jax_tvm_ffi
import numpy as np
import tvm_ffi.cpp
from jax import Array


def main() -> None:
    print("=" * 70)
    print("JAX-TVM-FFI: ctx.stream Example (GPU)")
    print("=" * 70)

    # Check if GPU is available
    try:
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            print("\nERROR: No GPU available. This example requires a GPU.")
            return
    except RuntimeError:
        print("\nERROR: No GPU backend available. This example requires a GPU.")
        return

    # Compile a CUDA kernel that receives the stream explicitly
    mod = tvm_ffi.cpp.load_inline(
        name="stream_example",
        cuda_sources=r"""
            #include <cuda_runtime.h>
            #include <cstdio>

            __global__ void vector_add_kernel(const float* x, const float* y, float* out, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                out[idx] = x[idx] + y[idx];
              }
            }

            void vector_add(tvm::ffi::TensorView x, tvm::ffi::TensorView y,
                           tvm::ffi::TensorView out, int64_t stream) {
              // Verify we received a valid GPU stream
              printf("Received GPU stream: 0x%llx\n", (unsigned long long)stream);
              TVM_FFI_ICHECK(stream != 0) << "GPU stream should be non-zero";

              // Validate inputs
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be 1D";
              TVM_FFI_ICHECK(y->ndim == 1) << "y must be 1D";
              TVM_FFI_ICHECK(out->ndim == 1) << "out must be 1D";
              TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have same shape";
              TVM_FFI_ICHECK(x->shape[0] == out->shape[0]) << "out must have same shape";

              int n = x->shape[0];
              int threads = 256;
              int blocks = (n + threads - 1) / threads;

              // Launch kernel on the stream provided by JAX
              cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
              vector_add_kernel<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const float*>(x->data),
                static_cast<const float*>(y->data),
                static_cast<float*>(out->data),
                n
              );

              // Check for launch errors
              cudaError_t err = cudaGetLastError();
              TVM_FFI_ICHECK(err == cudaSuccess)
                << "CUDA kernel launch failed: " << cudaGetErrorString(err);
            }
        """,
        functions=["vector_add"],
    )

    # Register with ctx.stream to receive the stream as a parameter
    jax_tvm_ffi.register_ffi_target(
        "example.vector_add",
        mod.vector_add,
        ["args", "rets", "ctx.stream"],  # ← ctx.stream passes stream as int64_t
        platform="gpu",
    )

    # Use the kernel
    print("\nRunning vector addition on GPU...")
    x = jnp.arange(1000, device=jax.devices("gpu")[0], dtype=jnp.float32)
    y = jnp.arange(1000, device=jax.devices("gpu")[0], dtype=jnp.float32) * 2

    @jax.jit
    def vector_add_ffi(x: Array, y: Array) -> Array:
        return jax.ffi.ffi_call(
            "example.vector_add",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
        )(x, y)

    result = vector_add_ffi(x, y)

    # Verify correctness
    expected = x + y
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    print(f"✓ Success! Computed {len(x)} elements")
    print(f"  x[0:5] = {x[:5]}")
    print(f"  y[0:5] = {y[:5]}")
    print(f"  result[0:5] = {result[:5]}")
    print(f"  expected[0:5] = {expected[:5]}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
