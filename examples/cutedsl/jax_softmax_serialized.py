# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JAX integration for serialized CuTe DSL softmax kernels.

Unlike ``jax_softmax``, this example does not keep a process-global TVM FFI
function registration. It dumps each TVM FFI compiled handle to in-memory
object bytes and embeds them in the StableHLO custom call. Each custom-call
instance loads an ORCJIT module and retains it with its resolved TVM FFI
function for the executable's lifetime.

Usage:
    python -m examples.cutedsl.jax_softmax_serialized
"""

from collections.abc import Callable

import cutlass
import jax
import jax.numpy as jnp
import jax_tvm_ffi
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tvm_ffi.cutlass import SerializedFunction, compile_to_object

from .jax_softmax import (
    DType,
    _get_cutlass_dtype,
    _make_example_tensor,
)
from .softmax import SoftmaxBackward, SoftmaxForward


def _compile_softmax_forward_object(
    dtype: type[cutlass.Numeric], N: int, M_example: int = 32
) -> SerializedFunction:
    example_x = _make_example_tensor(M_example, N, dtype)
    example_o = _make_example_tensor(M_example, N, dtype)
    return compile_to_object(SoftmaxForward(dtype, N), example_x, example_o)


def _compile_softmax_backward_object(
    dtype: type[cutlass.Numeric], N: int, M_example: int = 32
) -> SerializedFunction:
    example_dy = _make_example_tensor(M_example, N, dtype)
    example_y = _make_example_tensor(M_example, N, dtype)
    example_dx = _make_example_tensor(M_example, N, dtype)
    return compile_to_object(SoftmaxBackward(dtype, N), example_dy, example_y, example_dx)


_SERIALIZED_KERNELS: dict[tuple[str, int], Callable[[Array], Array]] = {}


def register_softmax_ops(N: int, dtype: DType = jnp.bfloat16) -> None:
    """Compile and serialize softmax kernels for one column count and dtype."""
    cache_key = (jnp.dtype(dtype).name, N)
    if cache_key in _SERIALIZED_KERNELS:
        return

    cutlass_dtype = _get_cutlass_dtype(dtype)
    forward = _compile_softmax_forward_object(cutlass_dtype, N)
    backward = _compile_softmax_backward_object(cutlass_dtype, N)
    forward_calls: dict[tuple[tuple[int, ...], str], Callable[[Array], Array]] = {}
    backward_calls: dict[tuple[tuple[int, ...], str], Callable[[Array, Array], Array]] = {}

    def get_forward_call(x: Array) -> Callable[[Array], Array]:
        key = (x.shape, jnp.dtype(x.dtype).name)
        if key not in forward_calls:
            forward_calls[key] = jax_tvm_ffi.ffi_call_from_serialized(
                forward,
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                platform="gpu",
                arg_spec=("args", "rets"),
                vmap_method="broadcast_all",
            )
        return forward_calls[key]

    def get_backward_call(g: Array) -> Callable[[Array, Array], Array]:
        key = (g.shape, jnp.dtype(g.dtype).name)
        if key not in backward_calls:
            backward_calls[key] = jax_tvm_ffi.ffi_call_from_serialized(
                backward,
                jax.ShapeDtypeStruct(g.shape, g.dtype),
                platform="gpu",
                arg_spec=("args", "rets"),
                vmap_method="broadcast_all",
            )
        return backward_calls[key]

    @jax.custom_vjp
    def softmax_fn(x: Array) -> Array:
        return get_forward_call(x)(x)

    def softmax_fwd(x: Array) -> tuple[Array, Array]:
        y = softmax_fn(x)
        return y, y

    def softmax_bwd(y: Array, g: Array) -> tuple[Array]:
        return (get_backward_call(g)(g, y),)

    softmax_fn.defvjp(softmax_fwd, softmax_bwd)
    _SERIALIZED_KERNELS[cache_key] = softmax_fn


def softmax(x: Array) -> Array:
    """Compute softmax along the last axis using serialized CuTe DSL kernels."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {x.shape}")

    key = (jnp.dtype(x.dtype).name, x.shape[-1])
    softmax_fn = _SERIALIZED_KERNELS.get(key)
    if softmax_fn is None:
        raise RuntimeError(
            f"Softmax not registered for dtype={x.dtype}, N={x.shape[-1]}. "
            f"Call register_softmax_ops(N={x.shape[-1]}, dtype={x.dtype}) first."
        )
    return softmax_fn(x)


def main() -> None:
    """Demonstrate serialized CuTe DSL softmax integration with JAX."""
    print("=" * 60)
    print("Serialized CuTe DSL Softmax + JAX Integration Demo")
    print("=" * 60)

    N, dtype = 1024, jnp.bfloat16
    print(f"\nConfig: N={N}, dtype={dtype}")

    print("\n[1] Compiling and serializing CuTe DSL kernels...")
    register_softmax_ops(N=N, dtype=dtype)
    print("    Done!")

    print("\n[2] Testing JIT compilation...")
    x = jax.random.normal(jax.random.key(42), (32, N), dtype=dtype)
    y = jax.jit(softmax)(x)
    y_ref = jax.nn.softmax(x, axis=-1)
    max_diff = float(jnp.abs(y - y_ref).max())
    assert max_diff < 1e-3, f"JIT failed: max_diff={max_diff}"
    print(f"    PASS (max_diff={max_diff:.2e})")

    print("\n[3] Testing autodiff (jax.grad)...")
    dx = jax.grad(lambda value: softmax(value).sum())(x)
    print(f"    PASS (dx shape={dx.shape})")

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
