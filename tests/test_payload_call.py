# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import pickle
import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax_tvm_ffi
import numpy
import pytest
import tvm_ffi
import tvm_ffi.cpp
from jax import numpy as jnp
from jax.experimental import serialize_executable

_PAYLOAD_CPP_SOURCE = r"""
    void add(tvm::ffi::Any increment, tvm::ffi::TensorView y,
             tvm::ffi::TensorView x, void* workspace) {
      TVM_FFI_ICHECK(workspace != nullptr);
      TVM_FFI_ICHECK_EQ(x.ndim(), 1);
      TVM_FFI_ICHECK_EQ(x.shape()[0], y.shape()[0]);
      for (int64_t i = 0; i < x.shape()[0]; ++i) {
        static_cast<float*>(y.data_ptr())[i] =
            static_cast<float*>(x.data_ptr())[i] + increment.cast<int64_t>();
      }
    }

    void subtract(tvm::ffi::Any decrement, tvm::ffi::TensorView y,
                  tvm::ffi::TensorView x, void* workspace) {
      TVM_FFI_ICHECK(workspace != nullptr);
      TVM_FFI_ICHECK_EQ(x.ndim(), 1);
      TVM_FFI_ICHECK_EQ(x.shape()[0], y.shape()[0]);
      for (int64_t i = 0; i < x.shape()[0]; ++i) {
        static_cast<float*>(y.data_ptr())[i] =
            static_cast<float*>(x.data_ptr())[i] - decrement.cast<int64_t>();
      }
    }
"""
_PAYLOAD_FUNCTIONS = ("add", "subtract")


def test_payload_target_registers_before_backend_initialization():
    script = """
import jax
import jax.numpy as jnp
import jax_tvm_ffi

call = jax_tvm_ffi.ffi_call_from_payload(
    b"payload",
    "missing.Loader",
    "missing",
    jax.ShapeDtypeStruct((1,), jnp.float32),
    platform="cpu",
)
try:
    jax.jit(call)(jnp.zeros((1,), jnp.float32))
except Exception as error:
    if "Payload loader 'missing.Loader' is not registered" not in str(error):
        raise
else:
    raise AssertionError("Missing payload loader unexpectedly succeeded")
"""
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


@pytest.mark.parametrize("object_bytes", [b"", bytearray(b"object"), "object", None])
def test_serialized_function_rejects_invalid_object_bytes(object_bytes):
    with pytest.raises(ValueError, match="object_bytes must be nonempty bytes"):
        jax_tvm_ffi.SerializedFunction(object_bytes, "add")


@pytest.mark.parametrize("function_name", ["", 1, None])
def test_serialized_function_rejects_invalid_function_name(function_name):
    with pytest.raises(ValueError, match="function_name must be a nonempty string"):
        jax_tvm_ffi.SerializedFunction(b"object", function_name)


@pytest.fixture(scope="module")
def payload_module(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("payload_module")
    module_path = tvm_ffi.cpp.build_inline(
        name="payload_module_test",
        cpp_sources=_PAYLOAD_CPP_SOURCE,
        functions=_PAYLOAD_FUNCTIONS,
        build_directory=str(tmp_path / "build"),
    )
    return tvm_ffi.load_module(module_path)


@pytest.fixture(scope="module")
def object_bytes(tmp_path_factory):
    pytest.importorskip("tvm_ffi_orcjit")
    tmp_path = tmp_path_factory.mktemp("payload_call")
    object_path = tvm_ffi.cpp.build_inline(
        name="payload_call_test",
        cpp_sources=_PAYLOAD_CPP_SOURCE,
        functions=_PAYLOAD_FUNCTIONS,
        build_directory=str(tmp_path / "build"),
        output="payload_call_test.o",
    )
    return Path(object_path).read_bytes()


def test_payload_call_caches_module_and_passes_workspace(payload_module, request):
    jax_tvm_ffi.clear_payload_module_cache()
    payload = b"payload-module-cache"
    loader_name = "jax_tvm_ffi_test.LoadPayloadModule"
    load_count = 0

    @tvm_ffi.register_global_func(loader_name, override=True)
    def load_module(payload_bytes):
        nonlocal load_count
        load_count += 1
        assert bytes(payload_bytes) == payload
        return payload_module

    def restore_loader():
        jax_tvm_ffi.clear_payload_module_cache()
        tvm_ffi.remove_global_func(loader_name)

    request.addfinalizer(restore_loader)

    call = jax_tvm_ffi.ffi_call_from_payload(
        payload,
        loader_name,
        "add",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        arg_spec=("attrs.increment", "rets", "args"),
        workspaces=(jax_tvm_ffi.Workspace(64),),
        vmap_method="broadcast_all",
    )
    add_two = jax.jit(lambda value: call(value, increment=2))
    add_three = jax.jit(lambda value: call(value, increment=3))
    subtract_call = jax_tvm_ffi.ffi_call_from_payload(
        payload,
        loader_name,
        "subtract",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        arg_spec=("attrs.decrement", "rets", "args"),
        workspaces=(jax_tvm_ffi.Workspace(64),),
        vmap_method="broadcast_all",
    )
    subtract_two = jax.jit(lambda value: subtract_call(value, decrement=2))
    x = jnp.arange(8, dtype=jnp.float32)

    stablehlo = str(add_two.lower(x).compiler_ir())
    assert "jax_tvm_ffi.payload_call" in stablehlo
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    assert load_count == 1
    numpy.testing.assert_equal(numpy.asarray(add_three(x)), numpy.asarray(x + 3))
    numpy.testing.assert_equal(numpy.asarray(subtract_two(x)), numpy.asarray(x - 2))
    assert load_count == 1
    assert jax_tvm_ffi.clear_payload_module_cache() == 1
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    assert load_count == 1

    add_four = jax.jit(lambda value: call(value, increment=4))
    numpy.testing.assert_equal(numpy.asarray(add_four(x)), numpy.asarray(x + 4))
    assert load_count == 2

    replacement_load_count = 0

    @tvm_ffi.register_global_func(loader_name, override=True)
    def replacement_load_module(payload_bytes):
        nonlocal replacement_load_count
        replacement_load_count += 1
        assert bytes(payload_bytes) == payload
        return payload_module

    add_five = jax.jit(lambda value: call(value, increment=5))
    numpy.testing.assert_equal(numpy.asarray(add_five(x)), numpy.asarray(x + 5))
    assert replacement_load_count == 1


def test_payload_cache_lifetime_follows_executable_owners(payload_module, request):
    jax_tvm_ffi.clear_payload_module_cache()
    payload = b"payload-module-lifetime"
    loader_name = "jax_tvm_ffi_test.LoadLifetimeModule"
    load_count = 0

    @tvm_ffi.register_global_func(loader_name)
    def load_module(payload_bytes):
        nonlocal load_count
        load_count += 1
        assert bytes(payload_bytes) == payload
        return payload_module

    def cleanup():
        jax_tvm_ffi.clear_payload_module_cache()
        tvm_ffi.remove_global_func(loader_name)

    request.addfinalizer(cleanup)
    call = jax_tvm_ffi.ffi_call_from_payload(
        payload,
        loader_name,
        "add",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        arg_spec=("attrs.increment", "rets", "args"),
        workspaces=(jax_tvm_ffi.Workspace(64),),
    )
    x = jnp.arange(8, dtype=jnp.float32)
    add_one = jax.jit(lambda value: call(value, increment=1)).lower(x).compile()
    add_two = jax.jit(lambda value: call(value, increment=2)).lower(x).compile()

    numpy.testing.assert_equal(numpy.asarray(add_one(x)), numpy.asarray(x + 1))
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    assert load_count == 1

    del add_one
    jax.clear_caches()
    gc.collect()

    add_three = jax.jit(lambda value: call(value, increment=3)).lower(x).compile()
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    numpy.testing.assert_equal(numpy.asarray(add_three(x)), numpy.asarray(x + 3))
    assert load_count == 1

    del add_two, add_three
    jax.clear_caches()
    gc.collect()

    add_four = jax.jit(lambda value: call(value, increment=4)).lower(x).compile()
    numpy.testing.assert_equal(numpy.asarray(add_four(x)), numpy.asarray(x + 4))
    assert load_count == 2


def test_object_call_loads_orcjit_object(object_bytes, request):
    request.addfinalizer(jax_tvm_ffi.clear_payload_module_cache)
    call = jax_tvm_ffi.ffi_call_from_object(
        object_bytes,
        "add",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        arg_spec=("attrs.increment", "rets", "args"),
        workspaces=(jax_tvm_ffi.Workspace(64),),
    )
    x = jnp.arange(8, dtype=jnp.float32)

    add_two = jax.jit(lambda value: call(value, increment=2))
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))


def test_object_executable_round_trip_in_fresh_process(object_bytes, tmp_path):
    call = jax_tvm_ffi.ffi_call_from_object(
        object_bytes,
        "add",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        arg_spec=("attrs.increment", "rets", "args"),
        workspaces=(jax_tvm_ffi.Workspace(64),),
    )
    x = jnp.arange(8, dtype=jnp.float32)
    compiled = jax.jit(lambda value: call(value, increment=2)).lower(x).compile()
    executable_path = tmp_path / "executable.pkl"
    executable_path.write_bytes(pickle.dumps(serialize_executable.serialize(compiled)))

    script = """
import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import jax_tvm_ffi
import numpy
import tvm_ffi_orcjit
from jax.experimental import serialize_executable

jax_tvm_ffi.register_payload_target(platform="cpu")
jax_tvm_ffi.register_payload_target(platform="cpu")
compiled = serialize_executable.deserialize_and_load(
    *pickle.loads(Path(sys.argv[1]).read_bytes()), backend="cpu"
)
x = jnp.arange(8, dtype=jnp.float32)
numpy.testing.assert_equal(numpy.asarray(compiled(x)), numpy.asarray(x + 2))
"""
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [sys.executable, "-c", script, str(executable_path)],
        check=True,
        env=environment,
        timeout=60,
    )


def test_object_entry_points_share_module_cache(object_bytes, request):
    jax_tvm_ffi.clear_payload_module_cache()
    loader_name = "jax_tvm_ffi.LoadOrcjitBase64ObjectModule"
    original_loader = tvm_ffi.get_global_func(loader_name)
    load_count = 0

    @tvm_ffi.register_global_func(loader_name, override=True)
    def tracked_loader(payload_bytes):
        nonlocal load_count
        load_count += 1
        return original_loader(payload_bytes)

    def cleanup():
        jax_tvm_ffi.clear_payload_module_cache()
        tvm_ffi.register_global_func(loader_name, original_loader, override=True)

    request.addfinalizer(cleanup)
    serialized = jax_tvm_ffi.SerializedFunction(object_bytes, "add")
    result = jax.ShapeDtypeStruct((8,), jnp.float32)
    call_options = {
        "platform": "cpu",
        "arg_spec": ("attrs.increment", "rets", "args"),
        "workspaces": (jax_tvm_ffi.Workspace(64),),
    }
    object_call = jax_tvm_ffi.ffi_call_from_object(object_bytes, "add", result, **call_options)
    serialized_call = jax_tvm_ffi.ffi_call_from_serialized(serialized, result, **call_options)
    x = jnp.arange(8, dtype=jnp.float32)
    add_one = jax.jit(lambda value: object_call(value, increment=1)).lower(x).compile()
    add_two = jax.jit(lambda value: serialized_call(value, increment=2)).lower(x).compile()

    numpy.testing.assert_equal(numpy.asarray(add_one(x)), numpy.asarray(x + 1))
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    assert load_count == 1


def test_serialized_function_call_reuses_digest(object_bytes, monkeypatch, request):
    request.addfinalizer(jax_tvm_ffi.clear_payload_module_cache)
    serialized = jax_tvm_ffi.SerializedFunction(object_bytes, "add")

    def reject_rehash(_):
        raise AssertionError("serialized object was rehashed")

    monkeypatch.setattr(jax_tvm_ffi, "hashlib", types.SimpleNamespace(sha256=reject_rehash))
    call = jax_tvm_ffi.ffi_call_from_serialized(
        serialized,
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        arg_spec=("attrs.increment", "rets", "args"),
        workspaces=(jax_tvm_ffi.Workspace(64),),
    )
    x = jnp.arange(8, dtype=jnp.float32)

    numpy.testing.assert_equal(
        numpy.asarray(jax.jit(lambda value: call(value, increment=2))(x)),
        numpy.asarray(x + 2),
    )


@pytest.mark.parametrize("payload", [b"!!!!", b"AAA", b"A==="])
def test_base64_object_loader_rejects_invalid_payload(payload):
    loader = tvm_ffi.get_global_func("jax_tvm_ffi.LoadOrcjitBase64ObjectModule")
    with pytest.raises(Exception, match="Invalid base64 object payload"):
        loader(payload)


def test_payload_call_rejects_missing_export(payload_module, request):
    payload = b"missing-export-payload"
    loader_name = "jax_tvm_ffi_test.LoadMissingExportModule"

    @tvm_ffi.register_global_func(loader_name)
    def load_module(payload_bytes):
        assert bytes(payload_bytes) == payload
        return payload_module

    def cleanup():
        jax_tvm_ffi.clear_payload_module_cache()
        tvm_ffi.remove_global_func(loader_name)

    request.addfinalizer(cleanup)
    call = jax_tvm_ffi.ffi_call_from_payload(
        payload,
        loader_name,
        "missing",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
    )

    with pytest.raises(Exception, match="does not export TVM FFI function 'missing'") as error:
        jax.jit(call)(jnp.arange(8, dtype=jnp.float32))
    assert "INVALID_ARGUMENT" in str(error.value)


def test_clear_during_payload_load_prevents_repopulation(payload_module, request):
    loader_name = "jax_tvm_ffi_test.LoadBlockingModule"
    payload = b"blocking-payload"
    load_started = threading.Event()
    allow_load = threading.Event()
    load_count = 0

    @tvm_ffi.register_global_func(loader_name)
    def load_module(payload_bytes):
        nonlocal load_count
        load_count += 1
        assert bytes(payload_bytes) == payload
        if load_count == 1:
            load_started.set()
            assert allow_load.wait(timeout=10)
        return payload_module

    def cleanup():
        allow_load.set()
        jax_tvm_ffi.clear_payload_module_cache()
        tvm_ffi.remove_global_func(loader_name)

    request.addfinalizer(cleanup)
    call = jax_tvm_ffi.ffi_call_from_payload(
        payload,
        loader_name,
        "add",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        arg_spec=("attrs.increment", "rets", "args"),
        workspaces=(jax_tvm_ffi.Workspace(64),),
    )
    x = jnp.arange(8, dtype=jnp.float32)
    add_one = jax.jit(lambda value: call(value, increment=1))

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_result = executor.submit(lambda: numpy.asarray(add_one(x)))
        assert load_started.wait(timeout=10)
        assert jax_tvm_ffi.clear_payload_module_cache() == 0
        allow_load.set()
        numpy.testing.assert_equal(first_result.result(timeout=30), numpy.asarray(x + 1))

    assert jax_tvm_ffi.clear_payload_module_cache() == 0
    add_two = jax.jit(lambda value: call(value, increment=2))
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    assert load_count == 2


def test_workspace_rejects_nonpositive_size():
    with numpy.testing.assert_raises_regex(ValueError, "must be positive"):
        jax_tvm_ffi.Workspace(0)
