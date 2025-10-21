# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import jax_tvm_ffi
import tvm_ffi.cpp


def main() -> None:
    # create an inline module that defines the function add_one_cpu
    mod: tvm_ffi.Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }
        """,
        functions=["add_one_cpu"],
    )
    # register the function with jax_tvm_ffi
    jax_tvm_ffi.register_ffi_target("example.add_one_cpu", mod.add_one_cpu, platform="cpu")

    # Run the JIT-compiled functions
    @jax.jit
    def add_one_jax(x: jnp.ndarray) -> jnp.ndarray:
        """JAX function that calls the 'add_one' C++ implementation."""
        return jax.ffi.ffi_call(
            "example.add_one_cpu",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    cpu = jax.devices("cpu")[0]
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32, device=cpu)
    # Now run the function
    output_array = add_one_jax(x)
    print(f"JAX call to 'add_one_cpu' completed. output {output_array}")


main()
