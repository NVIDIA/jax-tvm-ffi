# JAX TVM FFI

JAX TVM FFI is a library that enables seamless integration between JAX and TVM FFI,
allowing you to expose any function that is compatible with TVM FFI ABI to JAX.

## Installation

```bash
pip install .
```

## Quick Start

```bash
python run_example.py
```

## Testing

```bash
python -m pytest -vvs tests
```

## Usage

The API allows users to take a `tvm_ffi.Function` and connect it as a JAX FFI function.
The example below shows how to do this:

```python
import jax
import jax.numpy as jnp
import jax_tvm_ffi
import tvm_ffi.cpp

# Create an inline C++ module
mod = tvm_ffi.cpp.load_inline(
    name="example",
    cpp_sources="""
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

# Register the function with JAX
jax_tvm_ffi.register_ffi_target("example.add_one_cpu", mod.add_one_cpu, platform="cpu")

# Use in JAX with JIT compilation
@jax.jit
def add_one_jax(x):
    return jax.ffi.ffi_call(
        "example.add_one_cpu",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )(x)

# Run the function
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
result = add_one_jax(x)
print(f"Result: {result}")  # [2. 3. 4.]
```

### Custom Argument Specifications

You can customize how arguments are passed to your C++ functions:

```python
# Pass attributes as arguments
def my_function(eps, ret, input):
    # eps is passed as an attribute, x and y as tensors
    pass

jax_tvm_ffi.register_ffi_target(
    "my.function",
    my_function,
    arg_spec=["attrs.eps", "ret", "args"],  # eps from attrs, then rets, then args
    platform="cpu"
)

# Call with attributes
result = jax.ffi.ffi_call("my.function", output_shape)(x, y, eps=1e-5)
```

### Python Callback

Because `tvm_ffi` supports Python functions out of the box, you can use the same
mechanism to register a Python function into the JAX system.
This feature is helpful for creating test cases and debugging.

```python
import numpy as np

def process_tensor(x, y):
    # Convert to NumPy arrays for processing
    x_np = np.from_dlpack(x)
    y_np = np.from_dlpack(y)
    y_np[:] = x_np + 1

jax_tvm_ffi.register_ffi_target(
    "process.tensor",
    process_tensor,
    platform="cpu",
    # Enable owned tensor access so from_dlpack can be called
    pass_owned_tensor=True
)
```

## CuTeDSL Integration

For an example of integrating high-performance [CuTeDSL](https://github.com/NVIDIA/cutlass) kernels
with JAX (including JIT, autodiff, and multi-GPU support), see [examples/cutedsl/](examples/cutedsl/).

```bash
pip install jax-tvm-ffi[cutedsl]
python -m examples.cutedsl.jax_softmax
```
