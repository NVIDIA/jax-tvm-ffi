# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark serialized payload loading and JAX FFI dispatch overhead on CPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax_tvm_ffi
import numpy as np
import tvm_ffi
import tvm_ffi.cpp

_SOURCE_TEMPLATE = r"""
{padding}
void add(tvm::ffi::Any increment, tvm::ffi::TensorView y,
         tvm::ffi::TensorView x) {
  static_cast<float*>(y.data_ptr())[0] =
      static_cast<float*>(x.data_ptr())[0] + increment.cast<int64_t>();
}
"""
_FUNCTION_NAME = "add"
_DIRECT_TARGET = "jax_tvm_ffi.benchmark_payload_call.direct"


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _time_ns(operation: Callable[[], Any]) -> tuple[Any, int]:
    start = time.perf_counter_ns()
    result = operation()
    return result, time.perf_counter_ns() - start


def _build_object(build_directory: Path, object_padding_bytes: int) -> bytes:
    padding = ""
    if object_padding_bytes:
        random_bytes = hashlib.shake_256(b"jax-tvm-ffi payload benchmark").digest(
            object_padding_bytes
        )
        encoded = "\\x" + "\\x".join(f"{value:02x}" for value in random_bytes)
        padding = (
            "__attribute__((used)) static const unsigned char benchmark_padding[] =\n"
            f'    "{encoded}";'
        )
    source = _SOURCE_TEMPLATE.replace("{padding}", padding)
    object_path = tvm_ffi.cpp.build_inline(
        name="payload_call_benchmark_object",
        cpp_sources=source,
        functions=(_FUNCTION_NAME,),
        build_directory=str(build_directory / "object"),
        output="payload_call_benchmark.o",
    )
    return Path(object_path).read_bytes()


def _benchmark_orcjit_loads(
    object_bytes: bytes, samples: int
) -> tuple[tvm_ffi.Module, tvm_ffi.Function, float, float, float, float]:
    loader = tvm_ffi.get_global_func("jax_tvm_ffi.LoadOrcjitObjectModule")
    load_times: list[float] = []
    lookup_times: list[float] = []
    direct_module = None
    direct_function = None
    for sample in range(samples + 1):
        module, load_ns = _time_ns(lambda: loader(object_bytes))
        function, lookup_ns = _time_ns(lambda: module.get_function(_FUNCTION_NAME))
        if function is None:
            raise RuntimeError(f"Loaded module does not export {_FUNCTION_NAME!r}")
        load_times.append(load_ns / 1_000)
        lookup_times.append(lookup_ns / 1_000)
        if sample == 0:
            direct_module = module
            direct_function = function
        else:
            del function, module
            gc.collect()
    assert direct_module is not None and direct_function is not None
    return (
        direct_module,
        direct_function,
        load_times[0],
        lookup_times[0],
        _median(load_times[1:]),
        _median(lookup_times[1:]),
    )


def _benchmark_call_factories(
    serialized: jax_tvm_ffi.SerializedFunction,
    result: jax.ShapeDtypeStruct,
    samples: int,
) -> tuple[float, float]:
    object_times = []
    prehashed_times = []
    for _ in range(samples):
        _, duration_ns = _time_ns(
            lambda: jax_tvm_ffi.ffi_call_from_object(
                serialized.object_bytes,
                serialized.function_name,
                result,
                platform="cpu",
                arg_spec=("attrs.increment", "rets", "args"),
            )
        )
        _, prehashed_ns = _time_ns(
            lambda: jax_tvm_ffi.ffi_call_from_serialized(
                serialized,
                result,
                platform="cpu",
                arg_spec=("attrs.increment", "rets", "args"),
            )
        )
        object_times.append(duration_ns / 1_000)
        prehashed_times.append(prehashed_ns / 1_000)
    return _median(object_times), _median(prehashed_times)


def _compile_once(
    call: Callable[..., jax.Array], x: jax.Array, increment: int
) -> tuple[Any, float, float]:
    lowered, lower_ns = _time_ns(
        lambda: jax.jit(lambda value: call(value, increment=increment)).lower(x)
    )
    executable, compile_ns = _time_ns(lowered.compile)
    return executable, lower_ns / 1_000_000, compile_ns / 1_000_000


def _benchmark_compilation(
    direct_call: Callable[..., jax.Array],
    payload_call: Callable[..., jax.Array],
    x: jax.Array,
    samples: int,
) -> dict[str, float]:
    # Prime XLA compilation before collecting either path.
    _compile_once(direct_call, x, increment=-1)

    direct_lower_ms = []
    direct_compile_ms = []
    for increment in range(samples):
        executable, lower_ms, compile_ms = _compile_once(direct_call, x, increment)
        direct_lower_ms.append(lower_ms)
        direct_compile_ms.append(compile_ms)
        del executable

    miss_lower_ms = []
    miss_compile_ms = []
    for increment in range(samples, 2 * samples):
        if jax_tvm_ffi.clear_payload_module_cache() != 0:
            raise RuntimeError("A payload executable unexpectedly outlived its miss sample")
        executable, lower_ms, compile_ms = _compile_once(payload_call, x, increment)
        miss_lower_ms.append(lower_ms)
        miss_compile_ms.append(compile_ms)
        del executable
        jax.clear_caches()
        gc.collect()

    jax_tvm_ffi.clear_payload_module_cache()
    anchor, _, _ = _compile_once(payload_call, x, increment=2 * samples)
    warm_lower_ms = []
    warm_compile_ms = []
    for increment in range(2 * samples + 1, 3 * samples + 1):
        executable, lower_ms, compile_ms = _compile_once(payload_call, x, increment)
        warm_lower_ms.append(lower_ms)
        warm_compile_ms.append(compile_ms)
        del executable
    del anchor
    jax.clear_caches()
    gc.collect()

    return {
        "direct_lower_ms": _median(direct_lower_ms),
        "direct_compile_ms": _median(direct_compile_ms),
        "payload_module_miss_lower_ms": _median(miss_lower_ms),
        "payload_module_miss_compile_ms": _median(miss_compile_ms),
        "payload_warm_lower_ms": _median(warm_lower_ms),
        "payload_warm_compile_ms": _median(warm_compile_ms),
    }


def _compile_loop(call: Callable[..., jax.Array], x: jax.Array, iterations: int) -> Any:
    def run(value: jax.Array) -> jax.Array:
        return jax.lax.fori_loop(
            0, iterations, lambda _, current: call(current, increment=1), value
        )

    return jax.jit(run).lower(x).compile()


def _benchmark_execution(
    direct_executable: Callable[[jax.Array], jax.Array],
    payload_executable: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    iterations: int,
    samples: int,
) -> tuple[float, float]:
    expected = np.asarray(x + iterations)
    np.testing.assert_equal(np.asarray(direct_executable(x)), expected)
    np.testing.assert_equal(np.asarray(payload_executable(x)), expected)
    direct_elapsed_ns = []
    payload_elapsed_ns = []
    for sample in range(samples):
        operations = (
            (direct_executable, direct_elapsed_ns),
            (payload_executable, payload_elapsed_ns),
        )
        if sample % 2:
            operations = tuple(reversed(operations))
        for executable, measurements in operations:
            _, duration_ns = _time_ns(lambda: executable(x).block_until_ready())
            measurements.append(duration_ns)
    return (
        _median(direct_elapsed_ns) / iterations,
        _median(payload_elapsed_ns) / iterations,
    )


def _benchmark_deserialization(
    direct_executable: Any, payload_executable: Any, samples: int
) -> dict[str, float | int]:
    direct_runtime = direct_executable.runtime_executable()
    payload_runtime = payload_executable.runtime_executable()
    direct_serialized = direct_runtime.serialize()
    payload_serialized = payload_runtime.serialize()
    client = payload_runtime.client
    devices = payload_runtime.local_devices()

    client.deserialize_executable(direct_serialized, devices)
    client.deserialize_executable(payload_serialized, devices)
    direct_times = []
    payload_hit_times = []
    for sample in range(samples):
        operations = (
            (direct_serialized, direct_times),
            (payload_serialized, payload_hit_times),
        )
        if sample % 2:
            operations = tuple(reversed(operations))
        for serialized_executable, measurements in operations:
            executable, duration_ns = _time_ns(
                lambda: client.deserialize_executable(serialized_executable, devices)
            )
            measurements.append(duration_ns / 1_000)
            del executable

    payload_forced_clear_times = []
    for _ in range(samples):
        jax_tvm_ffi.clear_payload_module_cache()
        executable, duration_ns = _time_ns(
            lambda: client.deserialize_executable(payload_serialized, devices)
        )
        payload_forced_clear_times.append(duration_ns / 1_000)
        del executable

    return {
        "direct_executable_bytes": len(direct_serialized),
        "payload_executable_bytes": len(payload_serialized),
        "direct_deserialize_us": _median(direct_times),
        "payload_cache_hit_deserialize_us": _median(payload_hit_times),
        "payload_forced_cache_clear_deserialize_us": _median(payload_forced_clear_times),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-samples", type=int, default=7)
    parser.add_argument("--execute-samples", type=int, default=25)
    parser.add_argument("--load-samples", type=int, default=15)
    parser.add_argument("--loop-iterations", type=int, default=20_000)
    parser.add_argument("--object-padding-bytes", type=int, default=0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if (
        min(
            args.compile_samples,
            args.execute_samples,
            args.load_samples,
            args.loop_iterations,
        )
        <= 0
    ):
        raise ValueError("All sample and iteration counts must be positive")
    if args.object_padding_bytes < 0:
        raise ValueError("object_padding_bytes must be nonnegative")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    importlib.import_module("tvm_ffi_orcjit")
    jax.config.update("jax_enable_compilation_cache", False)
    # JAX 0.11.1 replays queued FFI targets before queued FFI types at backend
    # initialization, so initialize before registering this stateful target.
    cpu_device = jax.devices("cpu")[0]
    with tempfile.TemporaryDirectory(prefix="jax_tvm_ffi_benchmark_") as temporary_directory:
        object_bytes = _build_object(Path(temporary_directory), args.object_padding_bytes)
        (
            _direct_module,
            direct_function,
            first_load_us,
            first_lookup_us,
            fresh_load_us,
            fresh_lookup_us,
        ) = _benchmark_orcjit_loads(object_bytes, args.load_samples)
        jax_tvm_ffi.register_ffi_target(
            _DIRECT_TARGET,
            direct_function,
            arg_spec=["attrs.increment", "rets", "args"],
            platform="cpu",
        )
        result = jax.ShapeDtypeStruct((1,), jnp.float32)
        direct_call = jax.ffi.ffi_call(_DIRECT_TARGET, result)
        serialized_construction_times = []
        for _ in range(args.load_samples):
            serialized, duration_ns = _time_ns(
                lambda: jax_tvm_ffi.SerializedFunction(object_bytes, _FUNCTION_NAME)
            )
            serialized_construction_times.append(duration_ns / 1_000)
        object_factory_us, prehashed_factory_us = _benchmark_call_factories(
            serialized, result, args.load_samples
        )
        payload_call = jax_tvm_ffi.ffi_call_from_serialized(
            serialized,
            result,
            platform="cpu",
            arg_spec=("attrs.increment", "rets", "args"),
        )
        x = jax.device_put(np.zeros((1,), np.float32), cpu_device)

        compilation = _benchmark_compilation(direct_call, payload_call, x, args.compile_samples)
        direct_loop = _compile_loop(direct_call, x, args.loop_iterations)
        payload_loop = _compile_loop(payload_call, x, args.loop_iterations)
        deserialization = _benchmark_deserialization(
            direct_loop, payload_loop, args.compile_samples
        )
        direct_ns, payload_ns = _benchmark_execution(
            direct_loop,
            payload_loop,
            x,
            args.loop_iterations,
            args.execute_samples,
        )

    print(f"platform: {jax.devices('cpu')[0].device_kind}")
    print(f"jax: {jax.__version__}")
    print(f"object_bytes: {len(object_bytes)}")
    print(f"serialized_function_construct_us: {_median(serialized_construction_times):.3f}")
    print(f"object_call_factory_us: {object_factory_us:.3f}")
    print(f"prehashed_call_factory_us: {prehashed_factory_us:.3f}")
    print(f"orcjit_first_module_add_us: {first_load_us:.3f}")
    print(f"orcjit_first_function_materialize_us: {first_lookup_us:.3f}")
    print(f"orcjit_warm_session_fresh_module_add_us: {fresh_load_us:.3f}")
    print(f"orcjit_warm_session_fresh_function_materialize_us: {fresh_lookup_us:.3f}")
    for name, value in compilation.items():
        print(f"{name}: {value:.3f}")
    for name, value in deserialization.items():
        print(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")
    print(f"direct_loop_amortized_ns_per_iteration: {direct_ns:.3f}")
    print(f"payload_loop_amortized_ns_per_iteration: {payload_ns:.3f}")
    print(f"payload_loop_overhead_ns_per_iteration: {payload_ns - direct_ns:.3f}")


if __name__ == "__main__":
    main()
