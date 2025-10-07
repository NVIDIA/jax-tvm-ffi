# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import jax
import jax_tvm_ffi
import numpy
import tvm_ffi.cpp
from jax import numpy as jnp


def test_add_one_cpu():
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";
              for (int i = 0; i < x->shape[0]; ++i) {
                static_cast<float*>(y->data)[i] = static_cast<float*>(x->data)[i] + 1;
              }
            }
        """,
        functions=["add_one_cpu"],
    )
    jax_tvm_ffi.register_ffi_target("add_one_cpu", mod.add_one_cpu, platform="cpu")
    x = jnp.arange(10, device=jax.devices("cpu")[0], dtype=jnp.float32)

    @jax.jit
    def add_one_cpu(x):
        return jax.ffi.ffi_call(
            "add_one_cpu",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    y = add_one_cpu(x)
    numpy.testing.assert_equal(numpy.array(x + 1), numpy.array(y))
