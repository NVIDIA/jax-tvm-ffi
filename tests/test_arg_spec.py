# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import jax
import jax_tvm_ffi
import numpy
import pytest
import tvm_ffi
from jax import numpy as jnp


def test_pass_attr():
    """Test passing value attributes to ffi functions"""
    call_counter = [0]
    expected_value = [None]
    expected_type = [None]

    def expect_value(value):
        call_counter[0] += 1
        assert value == expected_value[0]
        assert isinstance(value, expected_type[0])

    jax_tvm_ffi.register_ffi_target(
        "testing.expect_attr_value", expect_value, ["attrs.value"], platform="cpu"
    )
    # create a dummy cpu input to ensure call triggered on cpu
    x = jnp.arange(1, device=jax.devices("cpu")[0])

    def check_expect_value(input_type, value):
        old_call_counter = call_counter[0]
        expected_value[0] = value
        expected_type[0] = input_type
        jax.ffi.ffi_call(
            "testing.expect_attr_value", jax.ShapeDtypeStruct((), jnp.int32), has_side_effect=True
        )(x, value=value)
        assert call_counter[0] == old_call_counter + 1

    check_expect_value(bool, True)
    check_expect_value(int, 456)
    check_expect_value(float, 789.0)
    check_expect_value(str, "hello")
    check_expect_value(float, numpy.float32(123.0))


def test_pass_attr_list():
    """Test passing value attributes to ffi functions"""
    call_counter = [0]
    expected_value = [[1, 2, 3], [4.0, 5.0, 6.0]]

    def expect_value(list0, list1):
        call_counter[0] += 1
        assert tuple(list0) == tuple(expected_value[0])
        assert tuple(list1) == tuple(expected_value[1])

    jax_tvm_ffi.register_ffi_target(
        "testing.expect_attr_lists", expect_value, ["attrs.list0", "attrs.list1"], platform="cpu"
    )
    # create a dummy cpu input to ensure call triggered on cpu
    x = jnp.arange(1, device=jax.devices("cpu")[0])
    jax.ffi.ffi_call(
        "testing.expect_attr_lists", jax.ShapeDtypeStruct((), jnp.int32), has_side_effect=True
    )(x, list1=numpy.array(expected_value[1]), list0=numpy.array(expected_value[0]))
    assert call_counter[0] == 1


def test_pass_tensor_ret():
    """Test passing value attributes to ffi functions"""
    call_counter = [0]
    expected_eps = [None]
    expected_args = [None, None]
    expected_ret = [None]

    def pass_tensor_ret(eps, ret, arg0, arg1):
        """Callback to pass tensor return value"""
        call_counter[0] += 1
        assert eps == expected_eps[0]
        assert isinstance(arg0, tvm_ffi.Tensor)
        assert isinstance(arg1, tvm_ffi.Tensor)
        assert isinstance(ret, tvm_ffi.Tensor)
        assert arg0.shape == expected_args[0].shape
        assert arg0.dtype == expected_args[0].dtype
        assert arg1.shape == expected_args[1].shape
        assert arg1.dtype == expected_args[1].dtype
        assert ret.shape == expected_ret[0].shape
        assert ret.dtype == expected_ret[0].dtype

    jax_tvm_ffi.register_ffi_target(
        "testing.pass_tensor_ret", pass_tensor_ret, ["attrs.eps", "rets", "args"], platform="cpu"
    )
    # we create dummy inputs with different shapes/dtypes to ensure callback
    # matches the types/shape of the inputs and return value
    x = jnp.arange(1, device=jax.devices("cpu")[0])
    y = jnp.zeros((2, 3), device=jax.devices("cpu")[0], dtype=jnp.float32)
    z = jnp.zeros((3, 4), device=jax.devices("cpu")[0], dtype=jnp.float32)
    eps = 1.0
    expected_eps[0] = eps
    expected_args[0] = x
    expected_args[1] = y
    expected_ret[0] = z
    ret = jax.ffi.ffi_call(
        "testing.pass_tensor_ret",
        jax.ShapeDtypeStruct(z.shape, z.dtype),
    )(x, y, eps=eps)
    assert call_counter[0] == 1


def test_error_propagation():
    """Test error propagation from ffi functions"""

    def error_propagation():
        """Callback to propagate error"""
        raise ValueError("test error")

    jax_tvm_ffi.register_ffi_target(
        "testing.error_propagation", error_propagation, [], platform="cpu"
    )
    # create a dummy cpu input to ensure call triggered on cpu
    x = jnp.arange(1, device=jax.devices("cpu")[0])
    with pytest.raises(Exception, match="test error"):
        jax.ffi.ffi_call(
            "testing.error_propagation", jax.ShapeDtypeStruct((), jnp.int32), has_side_effect=True
        )(x)


def test_pass_owned_tensor():
    """Test passing owned tensor to ffi functions, we can use it to run dlpack compute :)"""

    def add_one_owned_tensor(x, y):
        assert isinstance(x, tvm_ffi.Tensor)
        assert isinstance(y, tvm_ffi.Tensor)
        xnp = numpy.from_dlpack(x)
        ynp = numpy.from_dlpack(y)
        ynp[:] = xnp + 1

    jax_tvm_ffi.register_ffi_target(
        "testing.add_one_owned_tensor", add_one_owned_tensor, platform="cpu", pass_owned_tensor=True
    )
    x = jnp.arange(100, device=jax.devices("cpu")[0])
    y = jax.ffi.ffi_call("testing.add_one_owned_tensor", jax.ShapeDtypeStruct(x.shape, x.dtype))(x)
    numpy.testing.assert_equal(numpy.array(x + 1), numpy.array(y))


def test_leak_owned_tensor_detection():
    """Test leaking owned tensor to ffi functions"""
    leak_retain = []

    def leak_owned_tensor(x):
        leak_retain.append(x)

    jax_tvm_ffi.register_ffi_target(
        "testing.leak_owned_tensor",
        leak_owned_tensor,
        ["args"],
        platform="cpu",
        pass_owned_tensor=True,
    )
    x = jnp.arange(100, device=jax.devices("cpu")[0])
    with pytest.raises(Exception, match="Leaked temp owned tensors"):
        jax.ffi.ffi_call("testing.leak_owned_tensor", jax.ShapeDtypeStruct(x.shape, x.dtype))(x)
