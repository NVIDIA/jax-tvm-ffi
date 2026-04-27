# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer GPU Kernels on JAX with the JAX TVM FFI Bridge.

Standalone script version of flashinfer_jax_tvm_ffi.ipynb.
Demonstrates the three-step pattern (build & load → register → call) for
three FlashInfer kernels: silu_and_mul, apply_rope, and single-request
decode attention, then composes all three inside a single @jax.jit region.

Requirements:
    pip install 'jax[cuda13]' flashinfer-python jax-tvm-ffi \
        --no-build-isolation \
        --extra-index-url https://flashinfer.ai/whl/cu130/

Usage:
    python flashinfer_jax_tvm_ffi.py
"""

import math
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jinja2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# GPU detection and environment setup
# ══════════════════════════════════════════════════════════════════════════════


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF/XLA info & warnings

if "CUDA_HOME" not in os.environ:
    try:
        nvcc = subprocess.check_output(["which", "nvcc"], text=True).strip()
        os.environ["CUDA_HOME"] = str(Path(nvcc).parent.parent)
    except subprocess.CalledProcessError:
        os.environ["CUDA_HOME"] = "/usr/local/cuda"

if "--xla_gpu_cuda_data_dir=" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_cuda_data_dir={os.environ['CUDA_HOME']}"
    ).strip()

import jax_tvm_ffi
from flashinfer.jit import env as jit_env
from flashinfer.jit import gen_act_and_mul_module, gen_jit_spec
from flashinfer.jit.attention.utils import generate_additional_params
from flashinfer.jit.rope import gen_rope_module
from flashinfer.jit.utils import write_if_different

print(f"JAX:        {jax.__version__}")
print(f"Devices:    {jax.devices()}")
print(f"CUDA home:  {os.environ.get('CUDA_HOME')}")
print(f"JIT cache:  {jit_env.FLASHINFER_GEN_SRC_DIR.parent}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Example 1: Gated SiLU (silu_and_mul)
# ══════════════════════════════════════════════════════════════════════════════

# Step 1 — Build & load
print("Compiling silu_and_mul (first run may take ~30 s)...")
silu_module = gen_act_and_mul_module("silu").build_and_load()
print(f"  Module type: {type(silu_module).__name__}")
print(f"  Function:    {silu_module.silu_and_mul}")


# Step 2 — Register
def _silu_and_mul_wrapper(out, x, enable_pdl):
    silu_module.silu_and_mul(out, x, enable_pdl)


jax_tvm_ffi.register_ffi_target(
    "flashinfer.silu_and_mul",
    _silu_and_mul_wrapper,
    arg_spec=["rets", "args", "attrs.enable_pdl"],
    platform="gpu",
    allow_cuda_graph=True,
    pass_owned_tensor=True,
)


# Step 3 — JAX-facing function
def silu_and_mul(x: jax.Array) -> jax.Array:
    """Fused silu(gate) * up.  Input: [..., 2H]  Output: [..., H]"""
    out_shape = (*x.shape[:-1], x.shape[-1] // 2)
    return jax.ffi.ffi_call(
        "flashinfer.silu_and_mul",
        jax.ShapeDtypeStruct(out_shape, x.dtype),
        vmap_method="broadcast_all",
    )(x, enable_pdl=False)


# Validate
TOKENS, HIDDEN = 32, 256
gate_up = jax.random.normal(jax.random.key(0), (TOKENS, 2 * HIDDEN), dtype=jnp.float16)
out = silu_and_mul(gate_up)
gate_ref = gate_up[..., :HIDDEN].astype(jnp.float32)
up_ref = gate_up[..., HIDDEN:].astype(jnp.float32)
ref = (jax.nn.silu(gate_ref) * up_ref).astype(jnp.float16)
np.testing.assert_allclose(
    np.array(out.astype(jnp.float32)),
    np.array(ref.astype(jnp.float32)),
    rtol=1e-2,
    atol=1e-2,
)
print(f"silu_and_mul: PASSED  ({gate_up.shape} → {out.shape})")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Example 2: Rotary Positional Embeddings (apply_rope)
# ══════════════════════════════════════════════════════════════════════════════

# Step 1 — Build & load
print("Compiling apply_rope...")
rope_module = gen_rope_module().build_and_load()
print(f"  Function: {rope_module.apply_rope}")


# Step 2 — Register (with argument reordering)
def _rope_wrapper(
    q_rope,
    k_rope,
    q,
    k,
    indptr,
    offsets,
    rotary_dim,
    interleave,
    rope_scale,
    rope_theta,
):
    rope_module.apply_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


jax_tvm_ffi.register_ffi_target(
    "flashinfer.apply_rope",
    _rope_wrapper,
    arg_spec=[
        "rets",
        "args",
        "attrs.rotary_dim",
        "attrs.interleave",
        "attrs.rope_scale",
        "attrs.rope_theta",
    ],
    platform="gpu",
    allow_cuda_graph=True,
    pass_owned_tensor=True,
)


# Step 3 — JAX-facing function
def apply_rope(q, k, indptr, offsets, *, rope_theta=1e4):
    """Apply rotary positional embeddings to packed query and key tensors."""
    head_dim = q.shape[-1]
    return jax.ffi.ffi_call(
        "flashinfer.apply_rope",
        (
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
        ),
        vmap_method="broadcast_all",
    )(
        q,
        k,
        indptr,
        offsets,
        rotary_dim=head_dim,
        interleave=False,
        rope_scale=1.0,
        rope_theta=float(rope_theta),
    )


# Validate
def _reference_rope(x, positions, theta=1e4):
    x32 = x.astype(jnp.float32)
    d = x32.shape[-1] // 2
    freqs = 1.0 / (theta ** (2.0 * jnp.arange(d, dtype=jnp.float32) / x32.shape[-1]))
    angles = positions[:, None].astype(jnp.float32) * freqs[None, :]
    cos_a = jnp.cos(angles)[:, None, :]
    sin_a = jnp.sin(angles)[:, None, :]
    x1, x2 = x32[..., :d], x32[..., d:]
    return jnp.concatenate([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1).astype(
        x.dtype
    )


NUM_HEADS, HEAD_DIM, SEQ_LEN, NUM_SEQ = 8, 64, 8, 2
ROPE_THETA = 1e4
q_in = jax.random.normal(
    jax.random.key(1), (NUM_SEQ * SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16
)
k_in = jax.random.normal(
    jax.random.key(2), (NUM_SEQ * SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16
)
indptr = jnp.array([0, SEQ_LEN, 2 * SEQ_LEN], dtype=jnp.int32)
offsets = jnp.array([0, 100], dtype=jnp.int32)
q_rot, k_rot = apply_rope(q_in, k_in, indptr, offsets, rope_theta=ROPE_THETA)
positions = jnp.concatenate([jnp.arange(SEQ_LEN, dtype=jnp.int32) + off for off in [0, 100]])
q_ref = _reference_rope(q_in, positions, theta=ROPE_THETA)
k_ref = _reference_rope(k_in, positions, theta=ROPE_THETA)
for name, got, want in [("q", q_rot, q_ref), ("k", k_rot, k_ref)]:
    np.testing.assert_allclose(
        np.array(got.astype(jnp.float32)),
        np.array(want.astype(jnp.float32)),
        rtol=1e-2,
        atol=1e-2,
    )
    max_err = float(jnp.max(jnp.abs(got.astype(jnp.float32) - want.astype(jnp.float32))))
    print(f"apply_rope {name}: PASSED  max_err={max_err:.5f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Example 3: Single-request decode attention
# ══════════════════════════════════════════════════════════════════════════════
# Type mappings for Jinja template rendering
DTYPE_CPP = {"float16": "half", "bfloat16": "nv_bfloat16", "float32": "float"}
DTYPE_SAFE = {"float16": "f16", "bfloat16": "bf16", "float32": "f32"}
POS_ENC = {
    0: "PosEncodingMode::kNone",
    1: "PosEncodingMode::kRoPELlama",
    2: "PosEncodingMode::kALiBi",
}


def gen_decode_jit_spec(dtype: str = "float16", head_dim: int = 64):
    """Return a JitSpec for type-specialized single-request decode attention."""
    s = DTYPE_SAFE[dtype]
    uri = (
        f"single_decode_with_kv_cache_dtype_q_{s}_dtype_kv_{s}_dtype_o_{s}_"
        f"head_dim_qk_{head_dim}_head_dim_vo_{head_dim}_"
        f"posenc_0_use_swa_False_use_logits_cap_False"
    )
    gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    gen_dir.mkdir(parents=True, exist_ok=True)

    params_decl, func_params, params_setter = generate_additional_params(
        additional_tensor_names=["maybe_alibi_slopes"],
        additional_tensor_dtypes=["float"],
        additional_scalar_names=[
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],
        additional_scalar_dtypes=["double", "double", "double", "double"],
    )

    kwargs = dict(
        additional_func_params=func_params,
        additional_params_decl=params_decl,
        additional_params_setter=params_setter,
        variant_decl="#include<flashinfer/attention/variants.cuh>",
        variant_name="DefaultAttention<false, false, false, false>",
        dtype_q=DTYPE_CPP[dtype],
        dtype_kv=DTYPE_CPP[dtype],
        dtype_o=DTYPE_CPP[dtype],
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        pos_encoding_mode=POS_ENC[0],
        use_sliding_window="false",
        use_logits_soft_cap="false",
    )

    csrc = jit_env.FLASHINFER_CSRC_DIR
    for tmpl, out_name in [
        ("single_decode_customize_config.jinja", "single_decode_config.inc"),
        ("single_decode_kernel_inst.jinja", "single_decode_kernel.cu"),
    ]:
        rendered = jinja2.Template((csrc / tmpl).read_text()).render(**kwargs)
        write_if_different(gen_dir / out_name, rendered)

    sources = [gen_dir / "single_decode_kernel.cu"]
    for fname in ["single_decode.cu", "single_decode_jit_binding.cu"]:
        dest = gen_dir / fname
        write_if_different(dest, (csrc / fname).read_text())
        sources.append(dest)

    return gen_jit_spec(uri, sources)


# Step 1 — Build & load
DTYPE, HEAD_DIM = "float16", 64
print(f"Compiling decode attention ({DTYPE}, head_dim={HEAD_DIM})...")
decode_module = gen_decode_jit_spec(DTYPE, HEAD_DIM).build_and_load()
print(f"  run function: {decode_module.run}")

# Step 2 — Register
_run = decode_module.run


def _decode_wrapper(
    out,
    tmp,
    lse_or_empty,
    q,
    k,
    v,
    alibi_or_empty,
    kv_layout_code,
    window_left,
    logits_soft_cap,
    sm_scale,
    rope_scale,
    rope_theta,
):
    lse = None if lse_or_empty.shape[0] == 0 else lse_or_empty
    alibi = None if alibi_or_empty.shape[0] == 0 else alibi_or_empty
    _run(
        q,
        k,
        v,
        tmp,
        out,
        lse,
        kv_layout_code,
        window_left,
        alibi,
        logits_soft_cap,
        sm_scale,
        rope_scale,
        rope_theta,
    )


DECODE_TARGET = f"flashinfer.single_decode_{DTYPE}_h{HEAD_DIM}"
jax_tvm_ffi.register_ffi_target(
    DECODE_TARGET,
    _decode_wrapper,
    arg_spec=[
        "rets",
        "args",
        "attrs.kv_layout_code",
        "attrs.window_left",
        "attrs.logits_soft_cap",
        "attrs.sm_scale",
        "attrs.rope_scale",
        "attrs.rope_theta",
    ],
    platform="gpu",
    allow_cuda_graph=True,
    pass_owned_tensor=True,
)


# Step 3 — JAX-facing function
def decode_attention(q, k, v):
    """Single-request GQA decode attention."""
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    tmp_elems = 32 * 1024 * 1024 // jnp.dtype(q.dtype).itemsize
    out, _, _ = jax.ffi.ffi_call(
        DECODE_TARGET,
        (
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((tmp_elems,), q.dtype),
            jax.ShapeDtypeStruct((0,), jnp.float32),
        ),
    )(
        q,
        k,
        v,
        jnp.empty((0,), dtype=jnp.float32),
        kv_layout_code=0,
        window_left=-1,
        logits_soft_cap=0.0,
        sm_scale=sm_scale,
        rope_scale=1.0,
        rope_theta=1e4,
    )
    return out


print(f"Registered '{DECODE_TARGET}'.")


# Validate
def _reference_gqa_decode(q, k, v):
    H_q, H_kv = q.shape[0], k.shape[1]
    scale = q.shape[-1] ** -0.5
    q32 = q.astype(jnp.float32).reshape(H_kv, H_q // H_kv, -1)
    scores = jnp.einsum("hgd,shd->hgs", q32, k.astype(jnp.float32)) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hgs,shd->hgd", weights, v.astype(jnp.float32))
    return out.reshape(H_q, -1)


NUM_QO, NUM_KV, KV_LEN = 16, 4, 512
q = jax.random.normal(jax.random.key(10), (NUM_QO, HEAD_DIM), dtype=jnp.float16)
k = jax.random.normal(jax.random.key(11), (KV_LEN, NUM_KV, HEAD_DIM), dtype=jnp.float16)
v = jax.random.normal(jax.random.key(12), (KV_LEN, NUM_KV, HEAD_DIM), dtype=jnp.float16)

out_raw = decode_attention(q, k, v)
out_ref = _reference_gqa_decode(q, k, v)
np.testing.assert_allclose(
    np.array(out_raw.astype(jnp.float32)), np.array(out_ref), rtol=1e-2, atol=1e-2
)
print(f"decode_attention: PASSED  ({NUM_QO}/{NUM_KV} GQA, kv_len={KV_LEN})")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Composing all three kernels in @jax.jit
# ══════════════════════════════════════════════════════════════════════════════

gate_up = jax.random.normal(jax.random.key(20), (4, 2 * HEAD_DIM), dtype=jnp.float16)
q_new = q.reshape(1, NUM_QO, HEAD_DIM)
k_new = k[:1]
indptr = jnp.array([0, 1], dtype=jnp.int32)
offsets = jnp.array([KV_LEN], dtype=jnp.int32)


@jax.jit
def decode_step(gate_up, q_new, k_new, k_cache, v_cache, indptr, offsets):
    """One LLM decode step compiled into a single XLA computation."""
    ffn_out = silu_and_mul(gate_up)
    q_r, k_r = apply_rope(q_new, k_new, indptr, offsets)
    attn_out = decode_attention(q_r.reshape(NUM_QO, HEAD_DIM), k_cache, v_cache)
    return ffn_out, attn_out


ffn_out, attn_out = decode_step(gate_up, q_new, k_new, k, v, indptr, offsets)

# Validate against calling each kernel individually (outside @jax.jit)
ffn_ref = silu_and_mul(gate_up)
q_r, k_r = apply_rope(q_new, k_new, indptr, offsets)
attn_ref = decode_attention(q_r.reshape(NUM_QO, HEAD_DIM), k, v)

np.testing.assert_allclose(
    np.array(ffn_out.astype(jnp.float32)),
    np.array(ffn_ref.astype(jnp.float32)),
    rtol=1e-2,
    atol=1e-2,
)
np.testing.assert_allclose(
    np.array(attn_out.astype(jnp.float32)),
    np.array(attn_ref.astype(jnp.float32)),
    rtol=1e-2,
    atol=1e-2,
)

print("@jax.jit composition: PASSED")
print(f"  gate_up {gate_up.shape} → ffn_out  {ffn_out.shape}")
print(f"  q_new   {q_new.shape}  → attn_out {attn_out.shape}")

# Latency benchmark
_ = decode_attention(q, k, v).block_until_ready()
N = 100
t0 = time.perf_counter()
for _ in range(N):
    decode_attention(q, k, v).block_until_ready()
us = (time.perf_counter() - t0) / N * 1e6
print(f"\ndecode_attention  kv_len={KV_LEN}, {NUM_QO}/{NUM_KV} GQA heads  →  {us:.1f} µs")
print("\nAll examples passed.")
