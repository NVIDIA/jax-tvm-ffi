# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sys
import types

import pytest
from jax_tvm_ffi import cutlass


@pytest.fixture
def fake_cutlass(monkeypatch):
    state = types.SimpleNamespace(
        calls=[],
        dump_error=None,
        has_gpu_module=True,
        object_bytes=b"object\0__tvm_ffi_softmax\0__tvm_ffi_module_init\0",
    )

    class CompiledFunction:
        function_name = "softmax"

        @property
        def has_gpu_module(self):
            return state.has_gpu_module

        def dump_to_object(self, function_prefix):
            state.calls.append(("dump_to_object", function_prefix))
            if state.dump_error is not None:
                raise state.dump_error
            return state.object_bytes

    class CompileCallable:
        def __getitem__(self, option):
            state.calls.append(("compile_option", option))
            return self

        def __call__(self, function, *args, options):
            state.calls.append(("compile", function, args, options))
            return CompiledFunction()

    enable_tvm_ffi = object()
    cutlass_dsl_module = types.ModuleType("cutlass.cutlass_dsl")
    cutlass_dsl_module.EnableTVMFFI = enable_tvm_ffi

    cute_module = types.ModuleType("cutlass.cute")
    cute_module.compile = CompileCallable()

    runtime_module = types.ModuleType("cutlass.runtime")

    def find_runtime_libraries(*, enable_tvm_ffi):
        state.calls.append(("find_runtime_libraries", enable_tvm_ffi))
        return ["/fake/libcute_dsl_runtime.so"]

    runtime_module.find_runtime_libraries = find_runtime_libraries

    cutlass_module = types.ModuleType("cutlass")
    cutlass_module.__path__ = []
    cutlass_module.cute = cute_module

    monkeypatch.setitem(sys.modules, "cutlass", cutlass_module)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute_module)
    monkeypatch.setitem(sys.modules, "cutlass.cutlass_dsl", cutlass_dsl_module)
    monkeypatch.setitem(sys.modules, "cutlass.runtime", runtime_module)
    monkeypatch.setattr(cutlass, "_RUNTIME_LIBRARY_HANDLES", {})
    monkeypatch.setattr(
        cutlass.ctypes,
        "CDLL",
        lambda path, *, mode: state.calls.append(("load_runtime", path, mode)) or object(),
    )
    state.enable_tvm_ffi = enable_tvm_ffi
    return state


def test_compile_to_object_dumps_tvm_ffi_compiled_handle(fake_cutlass):
    function = lambda: None

    result = cutlass.compile_to_object(
        function,
        "argument",
        gpu_arch="sm_90a",
        compile_options="--preserve-line-info --opt-level 2",
    )

    assert result == cutlass.SerializedFunction(
        object_bytes=fake_cutlass.object_bytes,
        function_name="softmax",
    )
    assert result.sha256 == hashlib.sha256(fake_cutlass.object_bytes).hexdigest()
    assert fake_cutlass.calls == [
        ("find_runtime_libraries", False),
        (
            "load_runtime",
            "/fake/libcute_dsl_runtime.so",
            getattr(cutlass.os, "RTLD_NOW", 0) | getattr(cutlass.os, "RTLD_GLOBAL", 0),
        ),
        ("compile_option", fake_cutlass.enable_tvm_ffi),
        (
            "compile",
            function,
            ("argument",),
            "--enable-tvm-ffi --gpu-arch sm_90a --preserve-line-info --opt-level 2",
        ),
        ("dump_to_object", "softmax"),
    ]


def test_compile_to_object_always_enables_tvm_ffi(fake_cutlass):
    cutlass.compile_to_object(lambda: None)

    compile_call = next(call for call in fake_cutlass.calls if call[0] == "compile")
    assert compile_call[-1] == "--enable-tvm-ffi"


def test_compile_to_object_appends_custom_options(fake_cutlass):
    cutlass.compile_to_object(
        lambda: None,
        gpu_arch="sm_90a",
        compile_options="--gpu-arch sm_100a --enable-tvm-ffi",
    )

    compile_call = next(call for call in fake_cutlass.calls if call[0] == "compile")
    assert compile_call[-1] == (
        "--enable-tvm-ffi --gpu-arch sm_90a --gpu-arch sm_100a --enable-tvm-ffi"
    )


def test_compile_to_object_loads_runtime_once_without_compilation_cache(fake_cutlass):
    function = lambda: None

    cutlass.compile_to_object(function)
    cutlass.compile_to_object(function)

    call_names = [call[0] for call in fake_cutlass.calls]
    assert call_names.count("find_runtime_libraries") == 1
    assert call_names.count("load_runtime") == 1
    assert call_names.count("compile") == 2
    assert call_names.count("dump_to_object") == 2


def test_compile_to_object_propagates_dump_failure(fake_cutlass):
    fake_cutlass.dump_error = RuntimeError("object emission failed")

    with pytest.raises(RuntimeError, match="object emission failed"):
        cutlass.compile_to_object(lambda: None)


def test_compile_to_object_rejects_generic_object_dump(fake_cutlass):
    fake_cutlass.object_bytes = b"object\0generic___tvm_ffi_softmax\0"

    with pytest.raises(RuntimeError, match="__tvm_ffi_module_init"):
        cutlass.compile_to_object(lambda: None)


def test_compile_to_object_rejects_cpu_only_handle(fake_cutlass):
    fake_cutlass.has_gpu_module = False

    with pytest.raises(ValueError, match="requires a CUDA CuTe DSL function"):
        cutlass.compile_to_object(lambda: None)

    assert all(call[0] != "dump_to_object" for call in fake_cutlass.calls)
