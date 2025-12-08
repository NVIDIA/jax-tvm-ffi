# CuTeDSL + JAX Softmax Example

High-performance softmax kernel written in [CuTeDSL](https://github.com/NVIDIA/cutlass), integrated with JAX via `jax-tvm-ffi`.

## Features

- **CuTeDSL kernels**: Forward and backward passes using online softmax algorithm
- **JIT compilation**: Works with `@jax.jit`
- **Autodiff**: Full gradient support via `jax.custom_vjp`
- **Multi-GPU**: Compatible with `jax.shard_map`

## Installation

```bash
pip install jax-tvm-ffi[cutedsl]
```

Or install dependencies separately:
```bash
pip install jax-tvm-ffi nvidia-cutlass-dsl
```

## Quick Start

```bash
python -m examples.cutedsl.jax_softmax
```

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
