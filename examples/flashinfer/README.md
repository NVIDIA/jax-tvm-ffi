# FlashInfer on JAX: Notebooks and Scripts

Two tutorials that show how to use FlashInfer GPU kernels from JAX via the jax-tvm-ffi bridge.

| File | What it covers |
|------|---------------|
| `flashinfer_jax_tvm_ffi.ipynb` / `.py` | The three-step bridge pattern (build & load, register, call) with three kernels: `silu_and_mul`, `apply_rope`, single-request decode attention |
| `gemma3_flashinfer_jax.ipynb` / `.py` | End-to-end Gemma 3 1B Instruct inference using FlashInfer kernels for prefill and decode |

Each tutorial is available as both a Jupyter notebook (with explanations) and a standalone Python script (for quick reading and running).

## Requirements

| Requirement | Details |
|-------------|---------|
| GPU | NVIDIA SM 7.5+ (Turing or later) |
| CUDA | 12.6+ |
| Python | 3.10+ |
| Container (recommended) | [NVIDIA NGC JAX container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax) |

## Installation

Recommended (CUDA 13):

```bash
# Core dependencies (both tutorials)
pip install 'jax[cuda13]'
pip install -U flashinfer-python jax-tvm-ffi \
    --no-build-isolation  \
    --extra-index-url https://flashinfer.ai/whl/cu130/

# Additional dependencies (Gemma 3 tutorial only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install safetensors huggingface_hub transformers
```

Replace `jax[cuda13]` with `jax[cuda12]` for CUDA 12.x.

Replace `cu130` with the appropriate variant for your [CUDA Toolkit version](https://developer.nvidia.com/cuda-toolkit-archive) (e.g., `cu126` for CUDA 12.6).

## Running

### Part 1: FlashInfer JAX TVM FFI bridge

As a notebook:

```bash
jupyter lab flashinfer_jax_tvm_ffi.ipynb
```

As a script:

```bash
python flashinfer_jax_tvm_ffi.py
```

The first run compiles three FlashInfer kernels (~30 s each). Subsequent runs use the cached `.so` files in `~/.cache/flashinfer/`.

### Part 2: Gemma 3 inference

Gemma 3 is a gated model. You must first:

1. Create a [Hugging Face](https://huggingface.co) account
2. Accept the Gemma 3 licence at [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
3. Authenticate using **one** of the methods below:

```bash
# Option A: environment variable (good for containers and CI)
export HF_TOKEN=hf_...

# Option B: persistent login (stores the token in ~/.cache/huggingface/token)
pip install huggingface_hub
huggingface-cli login
```

Then run:

```bash
# As a notebook
jupyter lab gemma3_flashinfer_jax.ipynb

# As a script
python gemma3_flashinfer_jax.py
```

If neither method is detected, the script will prompt you to paste your token interactively.

The first run downloads ~2 GB of model weights and compiles six FlashInfer kernels (gelu_tanh, rope, local/global decode, local/global prefill). Both are cached after the first run.

Note that this example grows the KV cache with `jnp.concatenate` during decode. This is intentional for this minimal demo because the single-request FlashInfer decode wrapper derives the current sequence length from the KV-cache tensor shape. This is not intended to show the most efficient JAX serving pattern. A production implementation would typically use a preallocated or paged KV cache, or pass sequence lengths/masks explicitly.

## What you'll learn

**Part 1** teaches the three-step pattern that every FlashInfer kernel follows:

```
Step 1  BUILD & LOAD   jit_spec.build_and_load()  ->  tvm_ffi.Module
Step 2  REGISTER       jax_tvm_ffi.register_ffi_target(name, wrapper, arg_spec)
Step 3  CALL           jax.ffi.ffi_call(name, output_shapes)(*inputs, **scalar_attrs)
```

Each example adds a new concept:

| Kernel | New concept |
|--------|------------|
| `silu_and_mul` | Minimal bridge: one input, one output, no argument reordering |
| `apply_rope` | Multiple outputs; argument reordering between JAX and TVM conventions |
| `single_decode` | Type-specialized JIT compilation; scratch buffers; optional-argument sentinels |

**Part 2** applies the same pattern to run Gemma 3 1B Instruct end-to-end, adding:

- `gelu_tanh_and_mul` (one-word change from `silu`)
- QK-norm (per-head RMSNorm on Q and K, new in Gemma 3)
- Dual RoPE theta (local layers use 10k, global layers use 1M)
- Local vs global attention with sliding window
- Prefill (parallel prompt processing) and decode (autoregressive generation)

## Troubleshooting

**`CUDA_HOME not found`** — Set it manually: `export CUDA_HOME=/usr/local/cuda`

**Compilation errors** — Delete the cache and retry: `rm -rf ~/.cache/flashinfer/`

**HF token errors** — Verify your token works: `huggingface-cli whoami`

**GPU interconnect warnings** — Harmless NVML messages on systems without NVLink. Suppressed by `TF_CPP_MIN_LOG_LEVEL=2` (set automatically in the scripts).
