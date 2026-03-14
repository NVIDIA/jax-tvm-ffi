# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for XLA_FFI_DataType_F64 (double precision float) support in JAX-TVM-FFI.

Covers:
- Tensor dtype passthrough (DecodeDataType: F64 → kDLFloat 64-bit)
- Scalar attribute passing (DecodeAttrScalar: double)
- Array attribute passing (DecodeAttrArray: double[])
- Actual computation on float64 tensors
"""

import os

import jax
import jax.numpy as jnp
import jax_tvm_ffi
import numpy
import pytest
import tvm_ffi.cpp

os.environ["JAX_PLATFORMS"] = "cpu"
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Shared C++ module
# ---------------------------------------------------------------------------

_mod = tvm_ffi.cpp.load_inline(
    name="f64_test",
    cpp_sources=r"""
        #include <cstdint>
        #include <dlpack/dlpack.h>

        // Validate that the FFI correctly maps F64 → kDLFloat 64-bit,
        // then copy input to output element-by-element.
        void validate_f64_dtype(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
            DLDataType f64_dtype{kDLFloat, 64, 1};
            TVM_FFI_ICHECK(input.dtype() == f64_dtype)
                << "Expected float64 dtype, got code=" << (int)input.dtype().code
                << " bits=" << (int)input.dtype().bits;
            TVM_FFI_ICHECK(input.ndim() == output.ndim()) << "ndim mismatch";
            for (int d = 0; d < input.ndim(); ++d) {
                TVM_FFI_ICHECK(input.size(d) == output.size(d)) << "shape mismatch at dim " << d;
            }
            int64_t n = input.size(0);
            const double* in_ptr  = static_cast<const double*>(input.data_ptr());
            double*       out_ptr = static_cast<double*>(output.data_ptr());
            for (int64_t i = 0; i < n; ++i) {
                out_ptr[i] = in_ptr[i];
            }
        }

        // Add a scalar double attribute to every element of input.
        void add_scalar_f64(tvm::ffi::Any eps_any,
                            tvm::ffi::TensorView input,
                            tvm::ffi::TensorView output) {
            double eps = eps_any.cast<double>();
            DLDataType f64_dtype{kDLFloat, 64, 1};
            TVM_FFI_ICHECK(input.dtype() == f64_dtype) << "Expected float64 input";
            TVM_FFI_ICHECK(input.size(0) == output.size(0)) << "Shape mismatch";
            const double* in_ptr  = static_cast<const double*>(input.data_ptr());
            double*       out_ptr = static_cast<double*>(output.data_ptr());
            for (int64_t i = 0; i < input.size(0); ++i) {
                out_ptr[i] = in_ptr[i] + eps;
            }
        }

        // Compute the dot product of input with a weight vector passed as an
        // attribute array (double[]).  Output is a 1-element float64 tensor.
        void dot_with_attr_array(tvm::ffi::Array<double> weights,
                                 tvm::ffi::TensorView input,
                                 tvm::ffi::TensorView output) {
            TVM_FFI_ICHECK_EQ((int64_t)weights.size(), input.size(0))
                << "weights length must match input length";
            const double* in_ptr  = static_cast<const double*>(input.data_ptr());
            double*       out_ptr = static_cast<double*>(output.data_ptr());
            double acc = 0.0;
            for (int64_t i = 0; i < input.size(0); ++i) {
                acc += in_ptr[i] * static_cast<double>(weights[i]);
            }
            out_ptr[0] = acc;
        }

        // 2-D float64 tensor: validate shape and strides, then copy.
        void validate_f64_2d(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
            DLDataType f64_dtype{kDLFloat, 64, 1};
            TVM_FFI_ICHECK(input.dtype() == f64_dtype) << "Expected float64 dtype";
            TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "Expected 2-D tensor";

            int64_t rows = input.size(0);
            int64_t cols = input.size(1);

            // Contiguous row-major: stride[0] == cols, stride[1] == 1
            TVM_FFI_ICHECK_EQ(input.stride(0), cols) << "Unexpected row stride";
            TVM_FFI_ICHECK_EQ(input.stride(1), 1)    << "Unexpected col stride";

            const double* in_ptr  = static_cast<const double*>(input.data_ptr());
            double*       out_ptr = static_cast<double*>(output.data_ptr());
            for (int64_t i = 0; i < rows * cols; ++i) {
                out_ptr[i] = in_ptr[i];
            }
        }
    """,
    functions=[
        "validate_f64_dtype",
        "add_scalar_f64",
        "dot_with_attr_array",
        "validate_f64_2d",
    ],
)

jax_tvm_ffi.register_ffi_target("f64.validate_dtype", _mod.validate_f64_dtype, platform="cpu")
jax_tvm_ffi.register_ffi_target(
    "f64.add_scalar", _mod.add_scalar_f64, ["attrs.eps", "args", "rets"], platform="cpu"
)
jax_tvm_ffi.register_ffi_target(
    "f64.dot_with_attr_array",
    _mod.dot_with_attr_array,
    ["attrs.weights", "args", "rets"],
    platform="cpu",
)
jax_tvm_ffi.register_ffi_target("f64.validate_2d", _mod.validate_f64_2d, platform="cpu")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_f64_dtype_passthrough():
    """F64 tensors are correctly decoded as kDLFloat 64-bit in C++."""
    cpu = jax.devices("cpu")[0]
    x = jnp.arange(8, dtype=jnp.float64, device=cpu)

    @jax.jit
    def copy(v):
        return jax.ffi.ffi_call(
            "f64.validate_dtype",
            jax.ShapeDtypeStruct(v.shape, v.dtype),
            vmap_method="broadcast_all",
        )(v)

    result = copy(x)
    assert result.shape == x.shape
    assert result.dtype == jnp.float64
    numpy.testing.assert_array_equal(numpy.array(result), numpy.array(x))


def test_f64_scalar_attr():
    """A float64 scalar attribute is decoded and applied to a float64 tensor."""
    cpu = jax.devices("cpu")[0]
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64, device=cpu)
    eps = 0.5

    @jax.jit
    def add_eps(v):
        return jax.ffi.ffi_call(
            "f64.add_scalar",
            jax.ShapeDtypeStruct(v.shape, v.dtype),
            vmap_method="broadcast_all",
        )(v, eps=eps)

    result = add_eps(x)
    numpy.testing.assert_allclose(numpy.array(result), numpy.array(x) + eps)


def test_f64_attr_array():
    """A double[] attribute array is passed and used in a dot-product computation."""
    cpu = jax.devices("cpu")[0]
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64, device=cpu)
    weights = numpy.array([0.5, 1.0, 2.0], dtype=numpy.float64)

    @jax.jit
    def dot(v):
        return jax.ffi.ffi_call(
            "f64.dot_with_attr_array",
            jax.ShapeDtypeStruct((1,), v.dtype),
            vmap_method="broadcast_all",
        )(v, weights=weights)

    result = dot(x)
    expected = float(numpy.dot(numpy.array(x), weights))
    numpy.testing.assert_allclose(float(numpy.array(result)[0]), expected, rtol=1e-12)


def test_f64_2d_shape_and_strides():
    """2-D float64 tensors have correct shape and contiguous strides."""
    cpu = jax.devices("cpu")[0]
    x = jnp.arange(12, dtype=jnp.float64, device=cpu).reshape(3, 4)

    @jax.jit
    def copy2d(v):
        return jax.ffi.ffi_call(
            "f64.validate_2d",
            jax.ShapeDtypeStruct(v.shape, v.dtype),
            vmap_method="broadcast_all",
        )(v)

    result = copy2d(x)
    assert result.shape == (3, 4)
    assert result.dtype == jnp.float64
    numpy.testing.assert_array_equal(numpy.array(result), numpy.array(x))


def test_f64_large_values():
    """float64 can represent values that would overflow float32."""
    cpu = jax.devices("cpu")[0]
    # 1e38 is close to float32 max (~3.4e38); 1e300 would overflow entirely
    x = jnp.array([1e100, -1e200, 1.7976931348623157e308], dtype=jnp.float64, device=cpu)

    @jax.jit
    def copy(v):
        return jax.ffi.ffi_call(
            "f64.validate_dtype",
            jax.ShapeDtypeStruct(v.shape, v.dtype),
            vmap_method="broadcast_all",
        )(v)

    result = copy(x)
    numpy.testing.assert_array_equal(numpy.array(result), numpy.array(x))


def test_f64_jit_multiple_calls():
    """JIT-compiled F64 kernel produces correct results across multiple invocations."""
    cpu = jax.devices("cpu")[0]

    @jax.jit
    def copy(v):
        return jax.ffi.ffi_call(
            "f64.validate_dtype",
            jax.ShapeDtypeStruct(v.shape, v.dtype),
            vmap_method="broadcast_all",
        )(v)

    for seed in [0, 42, 99]:
        rng = numpy.random.default_rng(seed)
        data = rng.standard_normal(16)
        x = jnp.array(data, dtype=jnp.float64, device=cpu)
        result = copy(x)
        numpy.testing.assert_allclose(numpy.array(result), data, rtol=1e-15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
