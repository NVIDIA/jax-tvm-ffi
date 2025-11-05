# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures and markers for jax-tvm-ffi tests."""

import jax
import pytest


def _has_gpu() -> bool:
    """Check if GPU is available without raising an exception."""
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


# Shared pytest marker for GPU tests
requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="Test requires a GPU")
