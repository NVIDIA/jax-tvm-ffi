# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JAX TVM FFI Python package."""

import hashlib
import importlib
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.ffi
import jax.numpy as jnp
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

_PAYLOAD_TARGET = "jax_tvm_ffi.payload_call"
_ORCJIT_OBJECT_LOADER = "jax_tvm_ffi.LoadOrcjitObjectModule"
_ORCJIT_REQUIRED_GLOBALS = (
    "tvm_ffi_orcjit.GlobalDefaultSession",
    "tvm_ffi_orcjit.SessionLoadModule",
)
_PAYLOAD_INTERNAL_ATTRS = frozenset(
    {
        "arg_spec",
        "device_type",
        "function_name",
        "num_workspace_outputs",
        "payload",
        "payload_loader",
        "payload_sha256",
    }
)
_payload_registration_lock = threading.Lock()
_registered_payload_targets: set[str] = set()

tvm_ffi.register_global_func(
    _ORCJIT_OBJECT_LOADER,
    _LIB.load_orcjit_object_module,
    override=True,
)


@dataclass(frozen=True)
class Workspace:
    """A hidden device buffer passed to a payload function as an opaque pointer."""

    size_in_bytes: int

    def __post_init__(self) -> None:
        if self.size_in_bytes <= 0:
            raise ValueError("Workspace size_in_bytes must be positive")


@dataclass(frozen=True)
class SerializedFunction:
    """Native object bytes, exported TVM FFI function name, and object digest."""

    object_bytes: bytes
    function_name: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.object_bytes, bytes) or not self.object_bytes:
            raise ValueError("object_bytes must be nonempty bytes")
        if not isinstance(self.function_name, str) or not self.function_name:
            raise ValueError("function_name must be a nonempty string")
        object.__setattr__(self, "sha256", hashlib.sha256(self.object_bytes).hexdigest())


def clear_payload_module_cache() -> int:
    """Invalidate process-wide payload-module lookup entries.

    Returns the number of reusable live lookup entries invalidated. The cache
    holds modules weakly, so module lifetime follows live JAX executables. Loads
    already in progress are not counted and cannot repopulate the cleared cache.
    """
    return int(_LIB.clear_payload_module_cache())


def _get_dl_device_type(platform: str) -> int:
    """Get the dl device type from the platform."""
    if platform == "cpu":
        return tvm_ffi.DLDeviceType.kDLCPU
    elif platform == "gpu":
        return tvm_ffi.DLDeviceType.kDLCUDA
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def register_payload_target(platform: str = "gpu") -> str:
    """Register the predefined target for loading payload-backed executables.

    Call this in a fresh process before deserializing a compiled executable or
    calling a deserialized JAX export. It registers the bridge and its state
    type without requiring the original object bytes or Python callable.
    Repeated registration for the same platform is safe. Returns the target
    name, which can also be used with ``jax.export.DisabledSafetyCheck.custom_call``.

    Import ``tvm_ffi_orcjit`` before loading object-backed executables. For a
    custom payload format, register its TVM global loader instead. Load any
    libraries required by the objects, such as the CuTe DSL runtime, as well.

    This initializes the selected JAX backend. Complete distributed
    initialization and backend configuration before calling this function.
    """
    device_type = int(_get_dl_device_type(platform))
    target = _PAYLOAD_TARGET
    with _payload_registration_lock:
        if platform in _registered_payload_targets:
            return target

    # JAX replays queued FFI targets before queued FFI types when a backend is
    # first created. Initialize it before registering this stateful target so
    # the type is visible when the handler registration is applied.
    jax.devices(platform)

    with _payload_registration_lock:
        if platform in _registered_payload_targets:
            return target

        jax.ffi.register_ffi_type(
            f"jax_tvm_ffi.payload_call_state.{platform}",
            {
                "type_id": jax.ffi.pycapsule(_LIB.payload_call_state_type_id(device_type)),
                "type_info": jax.ffi.pycapsule(_LIB.payload_call_state_type_info(device_type)),
            },
            platform=platform,
        )
        jax.ffi.register_ffi_target(
            target,
            {
                "instantiate": jax.ffi.pycapsule(
                    _LIB.payload_call_instantiate_handler(device_type)
                ),
                "execute": jax.ffi.pycapsule(_LIB.payload_call_execute_handler(device_type)),
            },
            platform=platform,
        )
        _registered_payload_targets.add(platform)
    return target


def _encode_arg_spec(arg_spec: Sequence[str] | None) -> tuple[str, frozenset[str]]:
    items = tuple(arg_spec) if arg_spec is not None else ("args", "rets")
    attr_names = set()
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValueError("arg_spec items must be nonempty strings")
        if "\0" in item:
            raise ValueError("arg_spec items cannot contain null characters")
        if item in ("args", "rets", "ctx.stream"):
            continue
        if not item.startswith("attrs.") or len(item) == len("attrs."):
            raise ValueError(
                f"Invalid arg spec {item!r}; expected 'args', 'rets', 'ctx.stream', "
                "or 'attrs.<key>'"
            )
        attr_name = item.removeprefix("attrs.")
        if attr_name in _PAYLOAD_INTERNAL_ATTRS:
            raise ValueError(f"arg_spec attribute {attr_name!r} is reserved")
        attr_names.add(attr_name)
    return "\0".join(items), frozenset(attr_names)


def _ffi_call_from_payload(  # noqa: PLR0913
    payload: bytes,
    payload_loader: str,
    function_name: str,
    result_shape_dtypes: Any,
    *,
    payload_sha256: str,
    platform: str = "gpu",
    arg_spec: Sequence[str] | None = None,
    workspaces: Sequence[Workspace] = (),
    vmap_method: str | None = None,
    input_layouts: Sequence[Any] | None = None,
    output_layouts: Sequence[Any] | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable[..., Any]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("payload must be nonempty bytes")
    if not payload_loader:
        raise ValueError("payload_loader must be nonempty")
    if not function_name:
        raise ValueError("function_name must be nonempty")
    if output_layouts is not None and len(output_layouts) != len(
        jax.tree.leaves(result_shape_dtypes)
    ):
        raise ValueError("output_layouts must describe only the visible results")
    encoded_arg_spec, expected_attr_names = _encode_arg_spec(arg_spec)

    target = register_payload_target(platform)
    result_leaves, result_tree = jax.tree.flatten(result_shape_dtypes)
    workspace_results = tuple(
        jax.ShapeDtypeStruct((workspace.size_in_bytes,), jnp.uint8) for workspace in workspaces
    )
    all_results = (*result_leaves, *workspace_results)
    all_output_layouts = None
    if output_layouts is not None:
        all_output_layouts = (*output_layouts, *((0,) for _ in workspaces))

    call = jax.ffi.ffi_call(
        target,
        all_results,
        vmap_method=vmap_method,
        input_layouts=input_layouts,
        output_layouts=all_output_layouts,
        input_output_aliases=input_output_aliases,
    )

    def wrapped(*args: Any, **attrs: Any) -> Any:
        if attrs.keys() != expected_attr_names:
            missing = sorted(expected_attr_names - attrs.keys())
            unexpected = sorted(attrs.keys() - expected_attr_names)
            raise ValueError(
                f"FFI attributes do not match arg_spec; missing={missing}, unexpected={unexpected}"
            )
        results = call(
            *args,
            **attrs,
            arg_spec=encoded_arg_spec,
            payload=payload,
            payload_loader=payload_loader,
            payload_sha256=payload_sha256,
            function_name=function_name,
            device_type=int(_get_dl_device_type(platform)),
            num_workspace_outputs=len(workspaces),
        )
        visible_results = tuple(results[: len(result_leaves)])
        return jax.tree.unflatten(result_tree, visible_results)

    return wrapped


def ffi_call_from_payload(  # noqa: PLR0913
    payload: bytes,
    payload_loader: str,
    function_name: str,
    result_shape_dtypes: Any,
    *,
    platform: str = "gpu",
    arg_spec: Sequence[str] | None = None,
    workspaces: Sequence[Workspace] = (),
    vmap_method: str | None = None,
    input_layouts: Sequence[Any] | None = None,
    output_layouts: Sequence[Any] | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable[..., Any]:
    """Build a JAX FFI call whose executable owns its compiled payload.

    ``payload_loader`` names a registered TVM global function with signature
    ``(Bytes) -> Module``. Modules are weakly interned by loader and payload
    SHA-256, then ``function_name`` is resolved from the shared module when an
    executable is instantiated. The module, resolved function, and decoded
    argument specification are owned by that executable.

    ``arg_spec`` uses the same ``args``, ``rets``, ``attrs.<key>``, and
    ``ctx.stream`` entries as :func:`register_ffi_target`. Keyword arguments to
    the returned callable must exactly match its ``attrs.<key>`` entries.

    Workspace buffers are appended to the custom call outputs, hidden from the
    returned JAX value, and passed to the loaded function as opaque pointers
    after its ordinary tensor arguments and results.

    Constructing the first payload-backed call for a platform initializes that
    JAX backend to satisfy state-type registration ordering. Complete distributed
    initialization and backend configuration before creating the call.
    """
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("payload must be nonempty bytes")
    return _ffi_call_from_payload(
        payload,
        payload_loader,
        function_name,
        result_shape_dtypes,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        platform=platform,
        arg_spec=arg_spec,
        workspaces=workspaces,
        vmap_method=vmap_method,
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        input_output_aliases=input_output_aliases,
    )


def _require_orcjit() -> None:
    try:
        importlib.import_module("tvm_ffi_orcjit")
    except ImportError as error:
        raise ImportError(
            "Object-backed calls require apache-tvm-ffi-orcjit; install jax-tvm-ffi[orcjit]"
        ) from error

    if any(
        tvm_ffi.get_global_func(name, allow_missing=True) is None
        for name in _ORCJIT_REQUIRED_GLOBALS
    ):
        raise RuntimeError(
            "The installed apache-tvm-ffi-orcjit does not support loading serialized object "
            "bytes into the default execution session"
        )


def ffi_call_from_object(
    object_bytes: bytes,
    function_name: str,
    result_shape_dtypes: Any,
    *,
    platform: str = "gpu",
    arg_spec: Sequence[str] | None = None,
    workspaces: Sequence[Workspace] = (),
    vmap_method: str | None = None,
    input_layouts: Sequence[Any] | None = None,
    output_layouts: Sequence[Any] | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable[..., Any]:
    """Build a JAX FFI call backed by an in-memory native object file.

    The object bytes are embedded directly in a StableHLO custom-call attribute.
    At executable instantiation, TVM-FFI ORCJIT loads them from memory. JAX TVM FFI
    weakly interns the module by object SHA-256 and resolves ``function_name``
    from it.
    """
    _require_orcjit()
    if not isinstance(object_bytes, bytes) or not object_bytes:
        raise ValueError("object_bytes must be nonempty bytes")
    return _ffi_call_from_payload(
        object_bytes,
        _ORCJIT_OBJECT_LOADER,
        function_name,
        result_shape_dtypes,
        payload_sha256=hashlib.sha256(object_bytes).hexdigest(),
        platform=platform,
        arg_spec=arg_spec,
        workspaces=workspaces,
        vmap_method=vmap_method,
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        input_output_aliases=input_output_aliases,
    )


def ffi_call_from_serialized(
    serialized_function: SerializedFunction,
    result_shape_dtypes: Any,
    *,
    platform: str = "gpu",
    arg_spec: Sequence[str] | None = None,
    workspaces: Sequence[Workspace] = (),
    vmap_method: str | None = None,
    input_layouts: Sequence[Any] | None = None,
    output_layouts: Sequence[Any] | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable[..., Any]:
    """Build a JAX FFI call from a prehashed native object.

    This is the preferred launcher for :func:`jax_tvm_ffi.cutlass.compile_to_object`
    results. It embeds the object in StableHLO like :func:`ffi_call_from_object`,
    while reusing the immutable digest already stored in ``serialized_function``.
    """
    if not isinstance(serialized_function, SerializedFunction):
        raise TypeError("serialized_function must be a SerializedFunction")
    _require_orcjit()
    return _ffi_call_from_payload(
        serialized_function.object_bytes,
        _ORCJIT_OBJECT_LOADER,
        serialized_function.function_name,
        result_shape_dtypes,
        payload_sha256=serialized_function.sha256,
        platform=platform,
        arg_spec=arg_spec,
        workspaces=workspaces,
        vmap_method=vmap_method,
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        input_output_aliases=input_output_aliases,
    )


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
