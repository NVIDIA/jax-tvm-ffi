# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an exported CuTe softmax without importing its kernel implementation."""

import argparse
import pickle
from pathlib import Path

import jax
import jax_tvm_ffi
import numpy as np
import tvm_ffi_orcjit  # noqa: F401 -- Registers the native object loader.
from jax import export
from jax.experimental import serialize_executable
from jax_tvm_ffi.cutlass import load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--format", choices=("export", "executable"), default="export")
    args = parser.parse_args()

    load_runtime()
    jax_tvm_ffi.register_payload_target(platform="gpu")
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        if args.format == "export":
            exported = export.deserialize(bytearray(args.artifact.read_bytes()))
            input_aval = exported.in_avals[0]
            input_spec = jax.ShapeDtypeStruct(input_aval.shape, input_aval.dtype)
            compiled = jax.jit(exported.call).lower(input_spec).compile()
        else:
            compiled = serialize_executable.deserialize_and_load(
                *pickle.loads(args.artifact.read_bytes()), backend="gpu"
            )

    x = np.random.default_rng(0).standard_normal((32, 1024)).astype(np.float32)
    actual = np.asarray(compiled(jax.device_put(x, gpu)))
    exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
    expected = exp_x / exp_x.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    print(f"PASS: {args.format} softmax matches the NumPy reference")


if __name__ == "__main__":
    main()
