# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUTLASS DSL compilation helpers for JAX TVM FFI."""

import ctypes
import importlib
import os
import threading
from collections.abc import Callable
from typing import Any

from . import SerializedFunction

_COMPILE_LOCK = threading.Lock()
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_LIBRARY_HANDLES: dict[str, ctypes.CDLL] = {}


def load_runtime() -> None:
    """Load CUTLASS DSL runtime libraries for ORCJIT symbol resolution.

    The handles remain alive for the process lifetime because serialized
    modules can run destructors after their last JAX executable is released.
    """
    with _RUNTIME_LOCK:
        if _RUNTIME_LIBRARY_HANDLES:
            return
        try:
            cutlass_runtime = importlib.import_module("cutlass.runtime")
        except ImportError as error:
            raise ImportError(
                "load_runtime requires CuTe DSL; install jax-tvm-ffi[cutedsl]"
            ) from error

        runtime_paths = cutlass_runtime.find_runtime_libraries(enable_tvm_ffi=False)
        if not runtime_paths:
            raise RuntimeError("CuTe DSL did not provide a runtime library")
        mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)
        handles: dict[str, ctypes.CDLL] = {}
        for path_like in runtime_paths:
            path = os.fspath(path_like)
            try:
                handles[path] = ctypes.CDLL(path, mode=mode)
            except OSError as error:
                raise RuntimeError(f"Failed to load CuTe DSL runtime library {path!r}") from error
        _RUNTIME_LIBRARY_HANDLES.update(handles)


def compile_to_object(
    function: Callable[..., None],
    *compile_args: Any,
    gpu_arch: str | None = None,
    compile_options: str | None = None,
) -> SerializedFunction:
    """Compile a CuTe DSL function to a serialized TVM FFI object.

    Args:
        function: A CUDA ``@cute.jit`` function or callable object.
        *compile_args: Representative arguments describing its runtime signature.
        gpu_arch: Optional CuTe target such as ``sm_90a``. When omitted, CuTe
            uses its normal target selection.
        compile_options: Additional ``cute.compile`` options. They are
            appended after the helper's defaults so they can override them.
            TVM FFI remains enabled.

    Returns:
        The object bytes and exported TVM FFI function name.

    Notes:
        This helper enables TVM FFI and serializes the compiled handle with
        its module initializer.
        Runtime libraries are loaded globally so ORCJIT can resolve the
        object's host-runtime symbols.
    """
    try:
        cute = importlib.import_module("cutlass.cute")
        cutlass_dsl = importlib.import_module("cutlass.cutlass_dsl")
    except ImportError as error:
        raise ImportError(
            "compile_to_object requires CuTe DSL; install jax-tvm-ffi[cutedsl]"
        ) from error

    options = ["--enable-tvm-ffi"]
    if gpu_arch:
        options.extend(("--gpu-arch", gpu_arch))
    if compile_options:
        options.append(compile_options)

    with _COMPILE_LOCK:
        load_runtime()
        compiled = cute.compile[cutlass_dsl.EnableTVMFFI](
            function,
            *compile_args,
            options=" ".join(options),
        )
        if not compiled.has_gpu_module:
            raise ValueError("compile_to_object requires a CUDA CuTe DSL function")
        function_name = compiled.function_name
        object_bytes = compiled.dump_to_object(function_name)
        if b"__tvm_ffi_module_init" not in object_bytes:
            raise RuntimeError(
                "CuTe DSL emitted an object without __tvm_ffi_module_init; "
                "install nvidia-cutlass-dsl==4.9.0.dev1 with TVM FFI object serialization support"
            )
        return SerializedFunction(
            object_bytes=object_bytes,
            function_name=function_name,
        )
