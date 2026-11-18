# CuTeDSL + JAX Softmax Example

High-performance softmax kernel written in [CuTeDSL](https://github.com/NVIDIA/cutlass), integrated with JAX via `jax-tvm-ffi`.

## Features

- **CuTeDSL kernels**: Forward and backward passes using online softmax algorithm
- **JIT compilation**: Works with `@jax.jit`
- **Autodiff**: Full gradient support via `jax.custom_vjp`
- **Multi-GPU**: Compatible with `jax.shard_map`

## Installation

From the repository root:

```bash
pip install -e ".[cutedsl]"
```

Or install dependencies separately:
```bash
pip install -e . "nvidia-cutlass-dsl==4.9.0.dev1"
```

## Quick Start

```bash
python -m examples.cutedsl.jax_softmax
```

To embed the compiled native objects in StableHLO instead of registering live
TVM FFI function handles, install the pinned ORCJIT 0.1.3 extra and run the serialized version:

```bash
pip install -e ".[cutedsl,orcjit]"
python -m examples.cutedsl.jax_softmax_serialized
```

CuTe DSL 4.9.0.dev1 and ORCJIT 0.1.3 are published on PyPI. The CuTe release lets
a CUDA TVM-FFI compiled handle dump itself to object bytes. The object
preserves the kernel's lazy CUDA initialization and exports
`__tvm_ffi_module_init` for explicit initialization.

The serialized example uses `cute.compile[EnableTVMFFI]` and serializes the
compiled handle into a `SerializedFunction` containing the TVM FFI object bytes,
exported function name, and SHA-256 digest. It embeds the bytes and name in the
HLO without writing an object file. The installed ORCJIT
extension loads object bytes through its existing execution-session API. JAX
TVM FFI caches the loaded module weakly by object SHA-256 and resolves each
requested function from it. Live JAX executables own the shared module. The
reusable entry point is `jax_tvm_ffi.cutlass.compile_to_object`. Pass additional
compiler flags in the `compile_options` string; they are appended after
the helper's defaults and can override them. The helper always adds
`--enable-tvm-ffi`. It also loads the installed CuTe runtime libraries with
process-global visibility so ORCJIT can resolve the object's runtime symbols;
call
`jax_tvm_ffi.cutlass.load_runtime()` when loading a previously serialized JAX
executable in a fresh process without recompiling the kernel first.

## Usage

```python
from examples.cutedsl.jax_softmax import softmax, register_softmax_ops
import jax
import jax.numpy as jnp

# Register kernels for your configuration
register_softmax_ops(N=1024, dtype=jnp.bfloat16)

# Create input
x = jax.random.normal(jax.random.key(0), (32, 1024), dtype=jnp.bfloat16)

# Use with JIT
y = jax.jit(softmax)(x)

# Use with grad
dx = jax.grad(lambda x: softmax(x).sum())(x)
```

### Multi-GPU with shard_map

```python
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = jax.devices("gpu")
if len(devices) >= 2:
    mesh = Mesh(np.array(devices[:2]), axis_names=('batch',))
    x_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', None)))
    y_sharded = jax.shard_map(
        softmax, mesh=mesh,
        in_specs=(P('batch', None),), out_specs=P('batch', None),
    )(x_sharded)
```

## Files

| File | Description |
|------|-------------|
| `softmax.py` | CuTeDSL kernel implementations (forward + backward) |
| `jax_softmax.py` | JAX integration of the softmax kernel |
| `jax_softmax_serialized.py` | JAX integration with object bytes embedded in StableHLO |
