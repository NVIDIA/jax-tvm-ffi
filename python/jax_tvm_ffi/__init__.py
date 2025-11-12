# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JAX TVM FFI Python package."""

import sys
from pathlib import Path
from typing import Callable, Optional

import jax.ffi
import tvm_ffi


def _load_lib() -> tvm_ffi.Module:
    # first look at the directory of the current file
    file_dir = Path(__file__).resolve().parent

    if sys.platform.startswith("win32"):
        lib_dll_name = "jax_tvm_ffi.dll"
    elif sys.platform.startswith("darwin"):
        lib_dll_name = "jax_tvm_ffi.dylib"
    else:
        lib_dll_name = "jax_tvm_ffi.so"

    dirs = [file_dir, file_dir / ".." / ".." / "build"]
    for dir in dirs:
        if (dir / lib_dll_name).exists():
            lib_path = dir / lib_dll_name
            return tvm_ffi.load_module(str(lib_path))

    raise RuntimeError(f"Cannot find library: {lib_dll_name}")


_LIB = _load_lib()


def _get_dl_device_type(platform: str) -> int:
    """Get the dl device type from the platform."""
    if platform == "cpu":
        return tvm_ffi.DLDeviceType.kDLCPU
    elif platform == "gpu":
        return tvm_ffi.DLDeviceType.kDLCUDA
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def register_ffi_target(
    name: str,
    function: tvm_ffi.Function,
    arg_spec: Optional[list[str]] = None,
    platform: str = "cpu",
    *,
    allow_cuda_graph: bool = False,
    pass_owned_tensor: bool = False,
    use_last_output_for_alloc_workspace: bool = False,
) -> Callable:
    """Function to register a ffi target for jax with tvm_ffi.Function

    Parameters
    ----------
    name: str
        The name of the ffi target

    function: tvm_ffi.Function
        The function to register

    arg_spec: Optional[list[str]]
        The arg spec of the function specifying how to map inputs to arguments to the function
        can be "args", "rets", or "attrs.<key>" for attributes

    platform: str
        The platform of the ffi target

    allow_cuda_graph: bool
        Whether the function can be used in cuda graph capture

    pass_owned_tensor: bool
        Whether the function can pass owned tensor to the function. This flag can be helpful
        in python callback when we want to further call from_dlpack on the tensor.
        However, the function should not retain the tensor after the function call.

    use_last_output_for_alloc_workspace: bool
        Whether to use the last output as allocation workspace for alloc_tensor() calls.
        If True, the user must provide workspace as the LAST element in result_shape_dtypes
        when calling the function. The workspace buffer should be jax.ShapeDtypeStruct((size,), jnp.uint8).

    Returns
    -------
    Callable
        The registered FFI capsule that can be used with jax.ffi.ffi_call

    Notes
    -----
    The arg_spec specifies how the inputs, outputs, and attributes are mapped to the
    underlying call to ffi function `fun`. Some examples:

    - `["args", "rets"]` maps to `fun(*args, *rets)`
    - `["attrs.key0", "args"]` maps to `fun(attrs["key0"], *args, *rets)`
    - `["attrs.key0", "args", "attrs.key1"]` maps to `fun(attrs["key0"], *args, attrs["key1"])`

    Workspace Allocation
    --------------------
    When use_last_output_for_alloc_workspace=True, the kernel can call alloc_tensor() for temporary memory.
    The user must explicitly provide workspace in the output tuple at call time:

    Example with workspace:
        >>> # 1. Register with workspace flag
        >>> my_kernel = register_ffi_target(
        ...     "my_kernel",
        ...     tvm_module.my_kernel,
        ...     use_last_output_for_alloc_workspace=True,
        ...     platform="gpu",
        ...     allow_cuda_graph=True
        ... )
        ...
        >>> # 2. Call with workspace in output tuple (LAST position)
        >>> workspace_size = 1024 * 4  # Compute based on your needs, each allocation is aligned to 128 bytes
        >>> result = jax.ffi.ffi_call(
        ...     "my_kernel",
        ...     (
        ...         jax.ShapeDtypeStruct(output_shape, jnp.float32),  # Actual output
        ...         jax.ShapeDtypeStruct((workspace_size,), jnp.uint8)  # Workspace (LAST!)
        ...     ),
        ...     input_array
        ... )
        >>> output = result[0]  # Strip workspace from results

    """
    # by default, we use the arg spec "args" and "rets"
    arg_spec = arg_spec if arg_spec is not None else ["args", "rets"]
    dl_device_type = _get_dl_device_type(platform)
    traits = 1 if allow_cuda_graph else 0
    fn = jax.ffi.pycapsule(
        _LIB.register_tvm_ffi_handler(
            function,
            arg_spec,
            dl_device_type,
            traits,
            pass_owned_tensor,
            use_last_output_for_alloc_workspace,
        )
    )
    jax.ffi.register_ffi_target(name, fn, platform=platform)
    return fn


def registered_count() -> int:
    """Get the number of registered functions"""
    return _LIB.registered_count()


def get_last_workspace_peak() -> int:
    """Get peak workspace usage from the last FFI call that used workspace.

    Returns
    -------
    int
        Peak workspace bytes used, or 0 if no workspace was used
    """
    return _LIB.get_last_workspace_peak()
