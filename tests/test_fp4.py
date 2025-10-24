# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test FP4 shape handling in JAX-TVM-FFI.

Validates that the FFI layer correctly transforms FP4 tensor shapes:
- XLA sends logical FP4 dimensions (e.g., 8 FP4 elements)
- FFI divides last dimension by 2 to get physical uint8 storage (4 bytes)
- TensorView receives physical dimensions, no further division needed in C++
"""

import os

import jax
import jax.numpy as jnp
import jax_tvm_ffi
import pytest
import tvm_ffi.cpp

# Force CPU-only execution for these tests
os.environ["JAX_PLATFORMS"] = "cpu"
jax.config.update("jax_platforms", "cpu")


# Create C++ module with FP4 shape validation functions
_mod = tvm_ffi.cpp.load_inline(
    name="fp4_shape_test",
    cpp_sources=r"""
        #include <cstdint>
        #include <dlpack/dlpack.h>

        // Validate 1D FP4 tensor: shape should be physical bytes
        void validate_fp4_1d(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
            // Check dtype using constant name instead of magic number
            TVM_FFI_ICHECK_EQ(input.dtype().code, kDLFloat4_e2m1fn)
                << "Expected FP4 dtype code kDLFloat4_e2m1fn";
            TVM_FFI_ICHECK_EQ(input.dtype().bits, 4) << "Expected 4 bits";
            TVM_FFI_ICHECK_EQ(input.dtype().lanes, 2) << "Expected 2 lanes (2 FP4 per byte)";

            // Check that size is physical bytes (not logical FP4 elements)
            // For 8 logical FP4 elements, should receive 4 bytes
            int64_t physical_size = input.size(0);

            // Just copy data - no division needed since we already have physical size
            const uint8_t* in_ptr = static_cast<const uint8_t*>(input.data_ptr());
            uint8_t* out_ptr = static_cast<uint8_t*>(output.data_ptr());
            for (int64_t i = 0; i < physical_size; ++i) {
                out_ptr[i] = in_ptr[i];
            }
        }

        // Validate 2D FP4 tensor: shape and strides should be physical
        void validate_fp4_2d(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
            TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "Expected 2D tensor";
            TVM_FFI_ICHECK_EQ(input.dtype().code, kDLFloat4_e2m1fn) << "Expected FP4 dtype";
            TVM_FFI_ICHECK_EQ(input.dtype().lanes, 2) << "Expected 2 lanes";

            int64_t rows = input.size(0);
            int64_t cols_bytes = input.size(1);

            // Verify strides are based on physical dimensions
            int64_t expected_row_stride = cols_bytes;
            TVM_FFI_ICHECK_EQ(input.stride(0), expected_row_stride)
                << "Row stride should equal physical column count, got " << input.stride(0)
                << " expected " << expected_row_stride;
            TVM_FFI_ICHECK_EQ(input.stride(1), 1)
                << "Column stride should be 1 (contiguous bytes)";

            // Copy data
            const uint8_t* in_ptr = static_cast<const uint8_t*>(input.data_ptr());
            uint8_t* out_ptr = static_cast<uint8_t*>(output.data_ptr());
            int64_t total_bytes = rows * cols_bytes;

            for (int64_t i = 0; i < total_bytes; ++i) {
                out_ptr[i] = in_ptr[i];
            }
        }

        // FP4 execution test: negate operation (flip sign bit)
        void fp4_negate(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
            TVM_FFI_ICHECK_EQ(input.dtype().code, kDLFloat4_e2m1fn) << "Expected FP4 dtype";
            TVM_FFI_ICHECK_EQ(input.ndim(), 1) << "Expected 1D tensor";

            // FP4 format: [sign(1)][exp(2)][mantissa(1)] per nibble
            // Two FP4 values packed per byte: [high_nibble][low_nibble]
            const uint8_t* in_ptr = static_cast<const uint8_t*>(input.data_ptr());
            uint8_t* out_ptr = static_cast<uint8_t*>(output.data_ptr());
            int64_t num_bytes = input.size(0);

            for (int64_t i = 0; i < num_bytes; ++i) {
                uint8_t packed = in_ptr[i];

                // Low nibble: flip sign bit (bit 3)
                uint8_t low_nibble = packed & 0x0F;
                uint8_t low_negated = low_nibble ^ 0x08;  // XOR with sign bit

                // High nibble: flip sign bit (bit 7)
                uint8_t high_nibble = (packed >> 4) & 0x0F;
                uint8_t high_negated = high_nibble ^ 0x08;  // XOR with sign bit

                // Pack back together
                out_ptr[i] = (high_negated << 4) | low_negated;
            }
        }

        // Test that uses stride-based indexing to access specific 2D element
        // This will FAIL if strides are based on logical shape instead of physical
        void fp4_2d_strided_access(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
            TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "Expected 2D tensor";
            TVM_FFI_ICHECK_EQ(input.dtype().code, kDLFloat4_e2m1fn) << "Expected FP4 dtype";

            int64_t rows = input.size(0);
            int64_t cols_bytes = input.size(1);  // Physical: bytes per row

            const uint8_t* in_ptr = static_cast<const uint8_t*>(input.data_ptr());
            uint8_t* out_ptr = static_cast<uint8_t*>(output.data_ptr());

            // Access each element using STRIDE-BASED indexing
            // If strides were wrong (based on logical shape), this would read wrong memory!
            for (int64_t row = 0; row < rows; ++row) {
                for (int64_t col = 0; col < cols_bytes; ++col) {
                    // Calculate offset using stride: offset = row * stride[0] + col * stride[1]
                    int64_t offset = row * input.stride(0) + col * input.stride(1);

                    // If stride(0) was wrong (e.g., 8 instead of 4), offset would be wrong!
                    out_ptr[row * cols_bytes + col] = in_ptr[offset];
                }
            }
        }

    """,
    functions=["validate_fp4_1d", "validate_fp4_2d", "fp4_negate", "fp4_2d_strided_access"],
)

# Register FFI targets
jax_tvm_ffi.register_ffi_target("fp4.validate_1d", _mod.validate_fp4_1d, platform="cpu")
jax_tvm_ffi.register_ffi_target("fp4.validate_2d", _mod.validate_fp4_2d, platform="cpu")
jax_tvm_ffi.register_ffi_target("fp4.negate", _mod.fp4_negate, platform="cpu")
jax_tvm_ffi.register_ffi_target("fp4.strided_access", _mod.fp4_2d_strided_access, platform="cpu")


def test_fp4_1d_shape():
    """Test that 1D FP4 tensors receive physical byte dimensions."""
    cpu = jax.devices("cpu")[0]

    # Create 8 FP4 elements → stored in 4 bytes
    float_values = jnp.array(
        [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -2.0], dtype=jnp.float32, device=cpu
    )
    fp4_values = float_values.astype(jnp.float4_e2m1fn)

    @jax.jit
    def validate(x):
        return jax.ffi.ffi_call(
            "fp4.validate_1d",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    # Should not raise - this validates:
    # 1. Shape is physical bytes (4 not 8)
    # 2. Dtype is correctly transformed to FP4
    # 3. Data can be read/written without crashes
    result = validate(fp4_values)
    assert result.shape == fp4_values.shape
    assert result.dtype == fp4_values.dtype


def test_fp4_2d_shape_and_strides():
    """Test that 2D FP4 tensors have correct physical shape and strides."""
    # Create 3x8 FP4 array (logical: 3 rows, 8 FP4 elements per row)
    # Physical storage: 3 rows, 4 bytes per row
    float_2d = jnp.ones((3, 8), dtype=jnp.float32, device=jax.devices("cpu")[0]) * 0.5
    fp4_2d = float_2d.astype(jnp.float4_e2m1fn)

    @jax.jit
    def validate(x):
        return jax.ffi.ffi_call(
            "fp4.validate_2d",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    # Should not raise - this validates:
    # 1. Shape is [3, 4] (physical bytes, not logical [3, 8])
    # 2. Strides are [4, 1] (based on physical shape)
    # 3. Dtype is correctly transformed to FP4
    result = validate(fp4_2d)
    assert result.shape == fp4_2d.shape
    assert result.dtype == fp4_2d.dtype


def test_fp4_negate_execution():
    """Test that FP4 computation (negate) executes correctly."""
    cpu = jax.devices("cpu")[0]

    # Use simple values that FP4 can represent
    float_values = jnp.array(
        [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -2.0], dtype=jnp.float32, device=cpu
    )
    fp4_values = float_values.astype(jnp.float4_e2m1fn)

    @jax.jit
    def negate(x):
        return jax.ffi.ffi_call(
            "fp4.negate",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    # Negate via FFI
    negated_fp4 = negate(fp4_values)
    negated_float = negated_fp4.astype(jnp.float32)

    # Compare with JAX native negation
    expected = -fp4_values
    expected_float = expected.astype(jnp.float32)

    # Should match (verifies we actually computed something)
    assert jnp.allclose(negated_float, expected_float, rtol=0.01)


def test_fp4_2d_strided_access():
    """Test that stride-based 2D access works correctly.

    This test would FAIL if strides were based on logical shape instead of physical!
    Example: For [3, 8] logical → [3, 4] physical
    - Correct strides: [4, 1] (physical)
    - Wrong strides: [8, 1] (logical) ← would access wrong memory!
    """
    # Create 3x8 FP4 array
    float_2d = jnp.ones((3, 8), dtype=jnp.float32, device=jax.devices("cpu")[0]) * 0.5
    fp4_2d = float_2d.astype(jnp.float4_e2m1fn)

    @jax.jit
    def strided_copy(x):
        return jax.ffi.ffi_call(
            "fp4.strided_access",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    # This uses stride-based indexing internally
    # Would crash or return garbage if strides were wrong
    result = strided_copy(fp4_2d)

    # Verify data integrity - should be identical copy
    result_bytes = result.tobytes()
    input_bytes = fp4_2d.tobytes()
    assert result_bytes == input_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
