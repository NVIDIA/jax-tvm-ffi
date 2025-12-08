# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
JAX Integration for CuTeDSL Softmax Kernels

This example demonstrates how to:
1. Compile CuTeDSL kernels with TVM FFI support
2. Register them with JAX via jax-tvm-ffi
3. Use custom_vjp for autodiff support
4. Work with JIT, grad, and shard_map

Usage:
    python -m examples.cutedsl.jax_softmax
"""

from typing import Any

import cutlass
import jax
import jax.numpy as jnp
import jax_tvm_ffi
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import EnableTVMFFI
from jax import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .softmax import SoftmaxBackward, SoftmaxForward

# Type alias for dtype
DType = Any  # jnp.dtype or numpy dtype

# Type mappings between JAX and CuTeDSL
_DTYPE_NAME_TO_CUTLASS = {
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
}

_CUTLASS_TO_JAX_DTYPE = {
    cutlass.Float16: jnp.float16,
    cutlass.BFloat16: jnp.bfloat16,
    cutlass.Float32: jnp.float32,
}


def _get_cutlass_dtype(jax_dtype: DType) -> type[cutlass.Numeric]:
    """Convert JAX/numpy dtype to CuTeDSL dtype."""
    dtype_name = jnp.dtype(jax_dtype).name
    cutlass_dtype = _DTYPE_NAME_TO_CUTLASS.get(dtype_name)
    if cutlass_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_name}. Use float16, bfloat16, or float32.")
    return cutlass_dtype


def _make_example_tensor(M: int, N: int, dtype: type[cutlass.Numeric]) -> Any:
    """Create an example tensor for kernel compilation."""
    jax_dtype = _CUTLASS_TO_JAX_DTYPE[dtype]
    gpu = jax.devices("gpu")[0]
    return from_dlpack(
        jnp.zeros((M, N), dtype=jax_dtype, device=gpu),
        assumed_align=16,
    ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))


def _compile_softmax_forward(dtype: type[cutlass.Numeric], N: int, M_example: int = 32) -> Any:
    """Compile the softmax forward kernel with TVM FFI support."""
    example_x = _make_example_tensor(M_example, N, dtype)
    example_o = _make_example_tensor(M_example, N, dtype)
    return cute.compile[EnableTVMFFI](SoftmaxForward(dtype, N), example_x, example_o)


def _compile_softmax_backward(dtype: type[cutlass.Numeric], N: int, M_example: int = 32) -> Any:
    """Compile the softmax backward kernel with TVM FFI support."""
    example_dy = _make_example_tensor(M_example, N, dtype)
    example_y = _make_example_tensor(M_example, N, dtype)
    example_dx = _make_example_tensor(M_example, N, dtype)
    return cute.compile[EnableTVMFFI](SoftmaxBackward(dtype, N), example_dy, example_y, example_dx)


# Cache: (dtype_name, N) -> (fwd_fn, bwd_fn, softmax_fn)
# fwd_fn/bwd_fn kept alive to prevent GC; softmax_fn is the custom_vjp wrapper
_REGISTERED_KERNELS: dict[tuple[str, int], tuple[Any, Any, Any]] = {}


def register_softmax_ops(N: int, dtype: DType = jnp.bfloat16) -> None:
    """Compile and register softmax kernels for a specific configuration.

    Must be called before using softmax() with that (N, dtype) combination.
    """
    cutlass_dtype = _get_cutlass_dtype(dtype)
    cache_key = (jnp.dtype(dtype).name, N)

    if cache_key in _REGISTERED_KERNELS:
        return  # Already registered

    dtype_str = str(cutlass_dtype).replace(".", "_")
    fwd_name = f"cute.softmax_fwd_{dtype_str}_n{N}"
    bwd_name = f"cute.softmax_bwd_{dtype_str}_n{N}"

    # Compile and register forward kernel
    fwd_fn = _compile_softmax_forward(cutlass_dtype, N)
    jax_tvm_ffi.register_ffi_target(fwd_name, fwd_fn, arg_spec=["args", "rets"], platform="gpu")

    # Compile and register backward kernel
    bwd_fn = _compile_softmax_backward(cutlass_dtype, N)
    jax_tvm_ffi.register_ffi_target(bwd_name, bwd_fn, arg_spec=["args", "rets"], platform="gpu")

    # Create custom_vjp wrapper for autodiff support
    @jax.custom_vjp
    def softmax_fn(x: Array) -> Array:
        return jax.ffi.ffi_call(
            fwd_name,
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    def softmax_fwd(x: Array) -> tuple[Array, Array]:
        y = softmax_fn(x)
        return y, y

    def softmax_bwd(y: Array, g: Array) -> tuple[Array]:
        dx = jax.ffi.ffi_call(
            bwd_name,
            jax.ShapeDtypeStruct(g.shape, g.dtype),
            vmap_method="broadcast_all",
        )(g, y)
        return (dx,)

    softmax_fn.defvjp(softmax_fwd, softmax_bwd)

    # Cache everything
    _REGISTERED_KERNELS[cache_key] = (fwd_fn, bwd_fn, softmax_fn)


def softmax(x: Array) -> Array:
    """Compute softmax along the last axis using CuTeDSL kernels.

    Args:
        x: Input array of shape (M, N)

    Returns:
        Softmax output of same shape

    Raises:
        ValueError: If input is not 2D
        RuntimeError: If kernels not registered for this configuration
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {x.shape}")

    key = (jnp.dtype(x.dtype).name, x.shape[-1])
    if key not in _REGISTERED_KERNELS:
        raise RuntimeError(
            f"Softmax not registered for dtype={x.dtype}, N={x.shape[-1]}. "
            f"Call register_softmax_ops(N={x.shape[-1]}, dtype={x.dtype}) first."
        )

    _, _, softmax_fn = _REGISTERED_KERNELS[key]
    return softmax_fn(x)


def main() -> None:
    """Demonstrate CuTeDSL softmax integration with JAX."""
    print("=" * 60)
    print("CuTeDSL Softmax + JAX Integration Demo")
    print("=" * 60)

    # Configuration
    N, dtype = 1024, jnp.bfloat16
    print(f"\nConfig: N={N}, dtype={dtype}")

    # Step 1: Compile and register kernels
    print("\n[1] Compiling CuTeDSL kernels with TVM FFI...")
    register_softmax_ops(N=N, dtype=dtype)
    print("    Done!")

    # Step 2: Test JIT compilation
    print("\n[2] Testing JIT compilation...")
    x = jax.random.normal(jax.random.key(42), (32, N), dtype=dtype)
    y = jax.jit(softmax)(x)
    y_ref = jax.nn.softmax(x, axis=-1)
    max_diff = float(jnp.abs(y - y_ref).max())
    assert max_diff < 1e-3, f"JIT failed: max_diff={max_diff}"
    print(f"    PASS (max_diff={max_diff:.2e})")

    # Step 3: Test autodiff
    print("\n[3] Testing autodiff (jax.grad)...")
    dx = jax.grad(lambda x: softmax(x).sum())(x)
    print(f"    PASS (dx shape={dx.shape})")

    # Step 4: Test shard_map (multi-GPU)
    print("\n[4] Testing shard_map...")
    devices = jax.devices("gpu")
    if len(devices) >= 2:
        mesh = Mesh(np.array(devices[:2]), axis_names=("batch",))
        x_sharded = jax.device_put(x, NamedSharding(mesh, P("batch", None)))
        y_sharded = jax.shard_map(
            softmax,
            mesh=mesh,
            in_specs=(P("batch", None),),
            out_specs=P("batch", None),
        )(x_sharded)
        max_diff = float(jnp.abs(y_sharded - y_ref).max())
        assert max_diff < 1e-3, f"shard_map failed: max_diff={max_diff}"
        print(f"    PASS (2 GPUs, max_diff={max_diff:.2e})")
    else:
        print(f"    SKIP (requires 2+ GPUs, found {len(devices)})")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
