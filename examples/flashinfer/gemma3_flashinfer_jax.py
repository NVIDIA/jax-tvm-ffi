# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3 on JAX with FlashInfer and the JAX TVM FFI Bridge.

Standalone script version of gemma3_flashinfer_jax.ipynb.
Runs end-to-end autoregressive inference for Gemma 3 1B Instruct using
FlashInfer kernels: gelu_tanh_and_mul, apply_rope, prefill attention,
and decode attention.

Requirements:
    pip install -U flashinfer-python jax-tvm-ffi \
        --no-build-isolation \
        --extra-index-url https://flashinfer.ai/whl/cu130/
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install safetensors huggingface_hub transformers

    Gemma 3 is a gated model. Authenticate via one of:
      - export HF_TOKEN=hf_...
      - huggingface-cli login

Usage:
    python gemma3_flashinfer_jax.py
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp

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
import torch as _torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

print(f"JAX:        {jax.__version__}")
print(f"Devices:    {jax.devices()}")
print(f"CUDA home:  {os.environ['CUDA_HOME']}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace authentication
# ══════════════════════════════════════════════════════════════════════════════

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    # Fall back to the token stored by `huggingface-cli login`
    try:
        from huggingface_hub import get_token

        HF_TOKEN = get_token() or ""
    except Exception:
        pass
if not HF_TOKEN:
    from getpass import getpass

    HF_TOKEN = getpass("HF_TOKEN not set. Enter your Hugging Face token: ")
if not HF_TOKEN:
    raise RuntimeError(
        "Hugging Face token is required. Either:\n"
        "  1. export HF_TOKEN=hf_...\n"
        "  2. huggingface-cli login\n"
        "  3. Enter it when prompted."
    )
os.environ["HF_TOKEN"] = HF_TOKEN

user_info = HfApi().whoami(token=HF_TOKEN)
print(f"Authenticated as: {user_info.get('name', 'Unknown')}")


# ══════════════════════════════════════════════════════════════════════════════
# Download model weights
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ID = "google/gemma-3-1b-it"
HF_CACHE = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, cache_dir=HF_CACHE)

print("Downloading model weights (~2 GB on first run)...")
model_dir = Path(snapshot_download(MODEL_ID, token=HF_TOKEN, cache_dir=str(HF_CACHE / "hub")))

index_path = model_dir / "model.safetensors.index.json"
if index_path.exists():
    with index_path.open() as f:
        shard_files = sorted(set(json.load(f)["weight_map"].values()))
else:
    shard_files = ["model.safetensors"]

print(f"Loading {len(shard_files)} shard(s) as JAX bfloat16 arrays...")
weights = {}
for shard in shard_files:
    with safe_open(str(model_dir / shard), framework="numpy") as f:
        for key in f.keys():
            weights[key] = jnp.array(f.get_tensor(key), dtype=jnp.bfloat16)

n_params = sum(int(w.size) for w in weights.values())
print(f"Loaded {len(weights)} tensors  ({n_params / 1e9:.2f} B parameters)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Model configuration
# ══════════════════════════════════════════════════════════════════════════════

with (model_dir / "config.json").open() as _f:
    _raw = json.load(_f)
cfg = _raw.get("text_config", _raw)

HIDDEN = cfg["hidden_size"]
INTERMEDIATE = cfg["intermediate_size"]
N_LAYERS = cfg["num_hidden_layers"]
N_Q = cfg["num_attention_heads"]
N_KV = cfg["num_key_value_heads"]
HEAD_DIM = cfg.get("head_dim", HIDDEN // N_Q)
VOCAB = cfg["vocab_size"]
RMS_EPS = cfg.get("rms_norm_eps", 1e-6)
SLIDING_WINDOW = cfg.get("sliding_window", 1024)
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
ROPE_THETA_LOCAL = int(cfg.get("rope_local_base_freq", 10_000))
ROPE_THETA_GLOBAL = int(cfg.get("rope_theta", 1_000_000))


def is_global(layer_idx: int) -> bool:
    return (layer_idx + 1) % 6 == 0


print(
    f"Architecture: hidden={HIDDEN}, layers={N_LAYERS}, "
    f"N_Q={N_Q}, N_KV={N_KV}, head_dim={HEAD_DIM} (GQA {N_Q // N_KV}x)"
)
print(
    f"Sliding window={SLIDING_WINDOW}, "
    f"rope_local={ROPE_THETA_LOCAL:,}, rope_global={ROPE_THETA_GLOBAL:,}"
)
print()


# ══════════════════════════════════════════════════════════════════════════════
# Compile and register all FlashInfer kernels
# ══════════════════════════════════════════════════════════════════════════════

from flashinfer.jit import (
    gen_act_and_mul_module,
    gen_single_decode_module,
    gen_single_prefill_module,
)
from flashinfer.jit.rope import gen_rope_module

# ── 1. gelu_tanh_and_mul ──────────────────────────────────────────────────────
print("Compiling gelu_tanh_and_mul...")
_gelu_mod = gen_act_and_mul_module("gelu_tanh").build_and_load()


def _gelu_wrapper(out, x, enable_pdl):
    _gelu_mod.gelu_tanh_and_mul(out, x, enable_pdl)


jax_tvm_ffi.register_ffi_target(
    "flashinfer.gelu_tanh_and_mul",
    _gelu_wrapper,
    arg_spec=["rets", "args", "attrs.enable_pdl"],
    platform="gpu",
    allow_cuda_graph=True,
    pass_owned_tensor=True,
)


def gelu_and_mul(x: jax.Array) -> jax.Array:
    out_shape = (*x.shape[:-1], x.shape[-1] // 2)
    return jax.ffi.ffi_call(
        "flashinfer.gelu_tanh_and_mul",
        jax.ShapeDtypeStruct(out_shape, x.dtype),
        vmap_method="broadcast_all",
    )(x, enable_pdl=False)


# ── 2. apply_rope ─────────────────────────────────────────────────────────────
print("Compiling apply_rope...")
_rope_mod = gen_rope_module().build_and_load()


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
    _rope_mod.apply_rope(
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


def apply_rope(q, k, indptr, offsets, rope_theta=1e4):
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
        rotary_dim=q.shape[-1],
        interleave=False,
        rope_scale=1.0,
        rope_theta=float(rope_theta),
    )


# ── 3. decode_attention: local + global variants ──────────────────────────────
_TMP_ELEMS = 32 * 1024 * 1024 // 2

print(f"Compiling decode attention (local, sliding-window={SLIDING_WINDOW})...")
_local_dec_mod = gen_single_decode_module(
    _torch.bfloat16,
    _torch.bfloat16,
    _torch.bfloat16,
    HEAD_DIM,
    HEAD_DIM,
    0,
    True,
    False,
).build_and_load()
print("Compiling decode attention (global, full attention)...")
_global_dec_mod = gen_single_decode_module(
    _torch.bfloat16,
    _torch.bfloat16,
    _torch.bfloat16,
    HEAD_DIM,
    HEAD_DIM,
    0,
    False,
    False,
).build_and_load()


def _make_decode_wrapper(run_fn):
    def _w(
        out,
        tmp,
        lse_or_empty,
        q,
        k,
        v,
        alibi_or_empty,
        layout,
        window_left,
        logits_soft_cap,
        sm_scale,
        rope_rcp_scale,
        rope_rcp_theta,
    ):
        lse = None if lse_or_empty.shape[0] == 0 else lse_or_empty
        alibi = None if alibi_or_empty.shape[0] == 0 else alibi_or_empty
        run_fn(
            q,
            k,
            v,
            tmp,
            out,
            lse,
            layout,
            window_left,
            alibi,
            logits_soft_cap,
            sm_scale,
            rope_rcp_scale,
            rope_rcp_theta,
        )

    return _w


_DEC_ARG_SPEC = [
    "rets",
    "args",
    "attrs.layout",
    "attrs.window_left",
    "attrs.logits_soft_cap",
    "attrs.sm_scale",
    "attrs.rope_rcp_scale",
    "attrs.rope_rcp_theta",
]
_KW = dict(platform="gpu", allow_cuda_graph=True, pass_owned_tensor=True)

jax_tvm_ffi.register_ffi_target(
    "flashinfer.decode_local",
    _make_decode_wrapper(_local_dec_mod.run),
    _DEC_ARG_SPEC,
    **_KW,
)
jax_tvm_ffi.register_ffi_target(
    "flashinfer.decode_global",
    _make_decode_wrapper(_global_dec_mod.run),
    _DEC_ARG_SPEC,
    **_KW,
)


def decode_attention(q, k_cache, v_cache, global_layer=False):
    target = "flashinfer.decode_global" if global_layer else "flashinfer.decode_local"
    window = -1 if global_layer else SLIDING_WINDOW
    out, _, _ = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((_TMP_ELEMS,), jnp.bfloat16),
            jax.ShapeDtypeStruct((0,), jnp.float32),
        ),
    )(
        q,
        k_cache,
        v_cache,
        jnp.empty((0,), dtype=jnp.float32),
        layout=0,
        window_left=window,
        logits_soft_cap=0.0,
        sm_scale=SM_SCALE,
        rope_rcp_scale=1.0,
        rope_rcp_theta=1.0,
    )
    return out


# ── 4. prefill_attention: local + global variants ─────────────────────────────
print(f"Compiling prefill attention (local, sliding-window={SLIDING_WINDOW})...")
_local_pre_mod = gen_single_prefill_module(
    "fa2",
    _torch.bfloat16,
    _torch.bfloat16,
    _torch.bfloat16,
    HEAD_DIM,
    HEAD_DIM,
    0,
    True,
    False,
    False,
).build_and_load()
print("Compiling prefill attention (global, full attention)...")
_global_pre_mod = gen_single_prefill_module(
    "fa2",
    _torch.bfloat16,
    _torch.bfloat16,
    _torch.bfloat16,
    HEAD_DIM,
    HEAD_DIM,
    0,
    False,
    False,
    False,
).build_and_load()


def _make_prefill_wrapper(run_fn):
    def _w(
        out,
        tmp,
        lse_or_empty,
        q,
        k,
        v,
        alibi_or_empty,
        mask_mode_code,
        layout,
        window_left,
        logits_soft_cap,
        sm_scale,
        rope_rcp_scale,
        rope_rcp_theta,
    ):
        lse = None if lse_or_empty.shape[0] == 0 else lse_or_empty
        alibi = None if alibi_or_empty.shape[0] == 0 else alibi_or_empty
        run_fn(
            q,
            k,
            v,
            tmp,
            out,
            lse,
            mask_mode_code,
            layout,
            window_left,
            None,  # maybe_packed_custom_mask
            alibi,  # maybe_alibi_slopes
            None,  # maybe_k_cache_sf
            None,  # maybe_v_cache_sf
            logits_soft_cap,
            sm_scale,
            rope_rcp_scale,
            rope_rcp_theta,
        )

    return _w


_PRE_ARG_SPEC = [
    "rets",
    "args",
    "attrs.mask_mode_code",
    "attrs.layout",
    "attrs.window_left",
    "attrs.logits_soft_cap",
    "attrs.sm_scale",
    "attrs.rope_rcp_scale",
    "attrs.rope_rcp_theta",
]

jax_tvm_ffi.register_ffi_target(
    "flashinfer.prefill_local",
    _make_prefill_wrapper(_local_pre_mod.run),
    _PRE_ARG_SPEC,
    **_KW,
)
jax_tvm_ffi.register_ffi_target(
    "flashinfer.prefill_global",
    _make_prefill_wrapper(_global_pre_mod.run),
    _PRE_ARG_SPEC,
    **_KW,
)


def prefill_attention(q, k, v, layer_i):
    glob = is_global(layer_i)
    target = "flashinfer.prefill_global" if glob else "flashinfer.prefill_local"
    window = -1 if glob else SLIDING_WINDOW
    out, _, _ = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((_TMP_ELEMS,), jnp.bfloat16),
            jax.ShapeDtypeStruct((0,), jnp.float32),
        ),
    )(
        q,
        k,
        v,
        jnp.empty((0,), dtype=jnp.float32),
        mask_mode_code=1,
        layout=0,
        window_left=window,
        logits_soft_cap=0.0,
        sm_scale=SM_SCALE,
        rope_rcp_scale=1.0,
        rope_rcp_theta=1.0,
    )
    return out


print("All kernels compiled and registered.")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Pure-JAX building blocks
# ══════════════════════════════════════════════════════════════════════════════


@jax.jit
def rms_norm(x, weight, eps=RMS_EPS):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32**2, axis=-1, keepdims=True) + eps)
    return y.astype(x.dtype) * (1.0 + weight)


@jax.jit
def qk_norm(x, weight):
    return rms_norm(x, weight)


def embed(token_ids):
    return weights["model.embed_tokens.weight"][token_ids] * math.sqrt(HIDDEN)


def lm_head(h):
    lm_w = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    return h.astype(jnp.float32) @ lm_w.astype(jnp.float32).T


def ffn(h, layer_i):
    pre = rms_norm(h, weights[f"model.layers.{layer_i}.pre_feedforward_layernorm.weight"])
    gate = pre @ weights[f"model.layers.{layer_i}.mlp.gate_proj.weight"].T
    up = pre @ weights[f"model.layers.{layer_i}.mlp.up_proj.weight"].T
    gate_up = jnp.concatenate([gate, up], axis=-1)
    hidden = gelu_and_mul(gate_up)
    out = hidden @ weights[f"model.layers.{layer_i}.mlp.down_proj.weight"].T
    out = rms_norm(out, weights[f"model.layers.{layer_i}.post_feedforward_layernorm.weight"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Prefill and decode layers
# ══════════════════════════════════════════════════════════════════════════════


def prefill_layer(h, layer_i):
    T = h.shape[0]
    glob = is_global(layer_i)
    rope_theta = ROPE_THETA_GLOBAL if glob else ROPE_THETA_LOCAL

    ln = rms_norm(h, weights[f"model.layers.{layer_i}.input_layernorm.weight"])
    q = (ln @ weights[f"model.layers.{layer_i}.self_attn.q_proj.weight"].T).reshape(
        T, N_Q, HEAD_DIM
    )
    k = (ln @ weights[f"model.layers.{layer_i}.self_attn.k_proj.weight"].T).reshape(
        T, N_KV, HEAD_DIM
    )
    v = (ln @ weights[f"model.layers.{layer_i}.self_attn.v_proj.weight"].T).reshape(
        T, N_KV, HEAD_DIM
    )

    q = qk_norm(q, weights[f"model.layers.{layer_i}.self_attn.q_norm.weight"])
    k = qk_norm(k, weights[f"model.layers.{layer_i}.self_attn.k_norm.weight"])

    indptr = jnp.array([0, T], dtype=jnp.int32)
    offsets = jnp.array([0], dtype=jnp.int32)
    q, k = apply_rope(q, k, indptr, offsets, rope_theta=rope_theta)

    attn_out = prefill_attention(q, k, v, layer_i)
    attn_out = attn_out.reshape(T, N_Q * HEAD_DIM)
    attn_out = attn_out @ weights[f"model.layers.{layer_i}.self_attn.o_proj.weight"].T
    attn_out = rms_norm(
        attn_out, weights[f"model.layers.{layer_i}.post_attention_layernorm.weight"]
    )
    h = h + attn_out
    h = h + ffn(h, layer_i)
    return h, (k, v)


def prefill(prompt_ids):
    h = embed(jnp.array(prompt_ids))
    kv_caches = []
    for i in range(N_LAYERS):
        h, kv_cache = prefill_layer(h, i)
        kv_caches.append(kv_cache)
    h_last = rms_norm(h[-1], weights["model.norm.weight"])
    return h_last, kv_caches


def decode_layer(h, layer_i, kv_cache, pos):
    glob = is_global(layer_i)
    rope_theta = ROPE_THETA_GLOBAL if glob else ROPE_THETA_LOCAL

    ln = rms_norm(h, weights[f"model.layers.{layer_i}.input_layernorm.weight"])
    q = (ln @ weights[f"model.layers.{layer_i}.self_attn.q_proj.weight"].T).reshape(N_Q, HEAD_DIM)
    k = (ln @ weights[f"model.layers.{layer_i}.self_attn.k_proj.weight"].T).reshape(N_KV, HEAD_DIM)
    v = (ln @ weights[f"model.layers.{layer_i}.self_attn.v_proj.weight"].T).reshape(N_KV, HEAD_DIM)

    q = qk_norm(q, weights[f"model.layers.{layer_i}.self_attn.q_norm.weight"])
    k = qk_norm(k, weights[f"model.layers.{layer_i}.self_attn.k_norm.weight"])

    q_pack, k_pack = q[None], k[None]
    indptr = jnp.array([0, 1], dtype=jnp.int32)
    offsets = jnp.array([pos], dtype=jnp.int32)
    q_r, k_r = apply_rope(q_pack, k_pack, indptr, offsets, rope_theta=rope_theta)
    q_r = q_r.squeeze(0)
    k_r = k_r.squeeze(0)

    # NOTE: Using jnp.concatenate to grow KV cache is intentional.
    # In standard JAX this is inefficient (O(N^2)) and you'd normally preallocate
    # and use lax.dynamic_update_slice. However, FlashInfer's single-request
    # decode kernel infers sequence length from k_cache/v_cache.shape.
    # Therefore we must keep the cache length equal to the actual number of tokens.
    # Switching to a fixed-size buffer would require a different FlashInfer API
    # (e.g. paged KV cache) or an explicit length/mask.
    k_cache, v_cache = kv_cache
    k_cache = jnp.concatenate([k_cache, k_r[None]], axis=0)
    v_cache = jnp.concatenate([v_cache, v[None]], axis=0)

    attn_out = decode_attention(q_r, k_cache, v_cache, global_layer=glob)
    attn_out = attn_out.reshape(N_Q * HEAD_DIM)
    attn_out = attn_out @ weights[f"model.layers.{layer_i}.self_attn.o_proj.weight"].T
    attn_out = rms_norm(
        attn_out, weights[f"model.layers.{layer_i}.post_attention_layernorm.weight"]
    )
    h = h + attn_out
    h = h + ffn(h, layer_i)
    return h, (k_cache, v_cache)


def decode_step(token_id, kv_caches, pos):
    h = embed(jnp.array([token_id])).squeeze(0)
    new_kv = []
    for i in range(N_LAYERS):
        h, kv = decode_layer(h, i, kv_caches[i], pos)
        new_kv.append(kv)
    h = rms_norm(h, weights["model.norm.weight"])
    logits = lm_head(h)
    return logits, new_kv


# ══════════════════════════════════════════════════════════════════════════════
# Text generation
# ══════════════════════════════════════════════════════════════════════════════

_STOP_IDS = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
for _tok in ["<end_of_turn>", "<eos>"]:
    _id = tokenizer.convert_tokens_to_ids(_tok)
    if _id is not None and _id != tokenizer.unk_token_id:
        _STOP_IDS.add(_id)


def generate(prompt, max_new_tokens=200, temperature=0.7, seed=0):
    """Autoregressive generation with the Gemma 3 instruct chat template."""
    messages = [{"role": "user", "content": prompt}]

    # Render chat template to plain text first.
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Then tokenize explicitly and extract only input_ids.
    enc = tokenizer(rendered, add_special_tokens=False)
    prompt_ids = enc["input_ids"]

    # Flatten batch dimension if present.
    if len(prompt_ids) > 0 and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]

    T = len(prompt_ids)
    key = jax.random.key(seed)

    print(f"Prompt ({T} tokens): {prompt!r}")
    print(f"Rendered prompt preview: {rendered[:120]!r}")

    print("Prefilling...", end=" ", flush=True)
    t0 = time.perf_counter()
    h_last, kv_caches = prefill(prompt_ids)
    jax.block_until_ready(h_last)
    print(f"{time.perf_counter() - t0:.1f}s")

    def _sample(logits, key):
        if temperature == 0.0:
            return int(jnp.argmax(logits)), key
        key, subkey = jax.random.split(key)
        return int(jax.random.categorical(subkey, logits / temperature)), key

    print("Response: ", end="", flush=True)

    generated = []
    for step in range(max_new_tokens):
        if step == 0:
            logits = lm_head(h_last)
        else:
            logits, kv_caches = decode_step(generated[-1], kv_caches, T + step - 1)
        next_tok, key = _sample(logits, key)
        generated.append(next_tok)
        if next_tok in _STOP_IDS:
            break
        print(tokenizer.decode([next_tok]), end="", flush=True)

    print()
    return tokenizer.decode(generated, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# Run inference
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    questions = [
        "What is the capital of Germany?",
        "How does rotary positional embedding differ from learned positional embedding?",
        "What is grouped-query attention and why is it useful?",
    ]
    for q in questions:
        generate(q, max_new_tokens=150, temperature=0.7, seed=0)
        print()
