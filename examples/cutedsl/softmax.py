# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CuTeDSL Softmax Kernels (Forward + Backward)

Single-CTA softmax using online algorithm (single pass for max + sum).
"""

import math
import operator
from collections.abc import Callable
from typing import Any, Optional

import cutlass
from cutlass import Float32, cute


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    """Warp-level reduction using shuffle instructions."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Block-level reduction using shared memory."""
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Row-wise reduction."""
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x

    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if cutlass.const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]

    val = warp_reduce(val, warp_op, width=min(threads_per_row, cute.arch.WARP_SIZE))

    if cutlass.const_expr(reduction_buffer is not None):
        warps_per_row = reduction_buffer.shape[1]
        if cutlass.const_expr(warps_per_row > 1):
            val = block_reduce(val, warp_op, reduction_buffer, init_val=init_val)

    return val


@cute.jit
def online_softmax_reduce(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
) -> tuple[Float32, Float32, cute.TensorSSA]:
    """Online softmax reduction - computes max, sum(exp), and exp(x - max) in one pass."""
    assert x.dtype == Float32, "x must be of type Float32"

    max_x = warp_reduce(
        x.reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
        cute.arch.fmax,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )

    log2_e = math.log2(math.e)
    exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)

    sum_exp_x = warp_reduce(
        exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )

    if cutlass.const_expr(reduction_buffer is not None):
        warps_per_row = reduction_buffer.shape[1]
        if cutlass.const_expr(warps_per_row > 1):
            lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
            row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row

            if lane_idx == 0:
                reduction_buffer[row_idx, col_idx] = _f32x2_to_i64(max_x, sum_exp_x)
            cute.arch.barrier()

            max_x_single_warp = -Float32.inf
            sum_exp_x_local = 0.0
            if lane_idx < warps_per_row:
                max_x_single_warp, sum_exp_x_local = _i64_to_f32x2(
                    reduction_buffer[row_idx, lane_idx]
                )

            max_x_final = warp_reduce(max_x_single_warp, cute.arch.fmax)
            sum_exp_x_local *= cute.math.exp(max_x_single_warp - max_x_final, fastmath=True)
            sum_exp_x = warp_reduce(sum_exp_x_local, operator.add)

            exp_x *= cute.math.exp(max_x - max_x_final, fastmath=True)
            max_x = max_x_final

    return max_x, sum_exp_x, exp_x


@cutlass.dsl_user_op
def _f32x2_to_i64(a: Float32, b: Float32, *, loc: Any = None, ip: Any = None) -> cutlass.Int64:
    """Pack two Float32 values into an Int64."""
    from cutlass._mlir.dialects import vector  # noqa: PLC0415
    from cutlass.cutlass_dsl import T  # noqa: PLC0415

    vec_f32x2 = vector.from_elements(
        T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip
    )
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    return cutlass.Int64(
        vector.extract(vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )


@cutlass.dsl_user_op
def _i64_to_f32x2(c: cutlass.Int64, *, loc: Any = None, ip: Any = None) -> tuple[Float32, Float32]:
    """Unpack an Int64 into two Float32 values."""
    from cutlass._mlir.dialects import vector  # noqa: PLC0415
    from cutlass.cutlass_dsl import T  # noqa: PLC0415

    vec_i64x1 = vector.from_elements(T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip)
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    res1 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return res0, res1


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    """Create predicate tensor for bounds checking along K dimension."""
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Fill out-of-bounds values in shared memory tensor."""
    tXrX_fill = cute.make_rmem_tensor_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if cutlass.const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


@cutlass.dsl_user_op
def domain_offset_i64(
    coord: cute.Coord, tensor: cute.Tensor, *, loc: Any = None, ip: Any = None
) -> cute.Tensor:
    """Apply coordinate offset to tensor using 64-bit arithmetic."""
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride), (
        "Coordinate and stride must have the same length"
    )
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


class SoftmaxForward:
    """Single-CTA Softmax Forward: y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))"""

    def __init__(self, dtype: type[cutlass.Numeric], N: int) -> None:
        self.dtype = dtype
        self.N = N

    def _calculate_threads_per_row(self) -> int:
        N = self.N
        if N <= 64:
            return 8
        elif N <= 128:
            return 16
        elif N <= 3072:
            return 32
        elif N <= 6144:
            return 64
        elif N <= 16384:
            return 128
        else:
            return 256

    def _get_num_threads(self) -> int:
        return 128 if self.N <= 16384 else 256

    def _get_tv_layout(self, num_copy_bits: int = 128) -> tuple[Any, Any]:
        vecsize = num_copy_bits // self.dtype.width
        assert self.N % vecsize == 0, f"N ({self.N}) must be divisible by vector size ({vecsize})"

        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0

        threads_per_row = self._calculate_threads_per_row()
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row)
        cols_per_block = num_threads // threads_per_row

        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    def _smem_size_in_bytes(self, tiler_mn: cute.Shape, num_warps: int) -> int:
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + num_warps * (cutlass.Int64.width // 8)
            + (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor) -> None:
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE

        self.kernel(mX, mO, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
        )

    @cute.kernel
    def kernel(
        self, mX: cute.Tensor, mO: cute.Tensor, tv_layout: cute.Layout, tiler_mn: cute.Shape
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        mX = domain_offset_i64((bidx * tiler_mn[0], 0), mX)
        mO = domain_offset_i64((bidx * tiler_mn[0], 0), mO)
        gX = cute.local_tile(mX, tiler_mn, (0, 0))
        gO = cute.local_tile(mO, tiler_mn, (0, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        threads_per_row = tv_layout.shape[0][0]
        warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
        reduction_buffer = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_ordered_layout((num_warps // warps_per_row, warps_per_row), order=(1, 0)),
            byte_alignment=8,
        )

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128
        )

        thr_copy_X = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX = cute.make_rmem_tensor_like(tXgX)
        tXrO = cute.make_rmem_tensor_like(tXgO)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1])
        tXpX = predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        if cutlass.const_expr(not is_even_N):
            fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        max_x, denom, exp_x = online_softmax_reduce(x, threads_per_row, reduction_buffer)
        y = exp_x * cute.arch.rcp_approx(denom)

        tXrO.store(y.to(tXrO.element_type))

        tOpO = predicate_k(thr_copy_O.partition_S(cX), limit=shape[1]) if not is_even_N else None

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tOpO)


class SoftmaxBackward:
    """Single-CTA Softmax Backward: dx_i = y_i * (dy_i - sum(dy_j * y_j))"""

    def __init__(self, dtype: type[cutlass.Numeric], N: int) -> None:
        self.dtype = dtype
        self.N = N

    def _calculate_threads_per_row(self) -> int:
        N = self.N
        if N <= 64:
            return 8
        elif N <= 128:
            return 16
        elif N <= 3072:
            return 32
        elif N <= 6144:
            return 64
        elif N <= 8192:
            return 128
        else:
            return 256

    def _get_num_threads(self) -> int:
        return 128 if self.N <= 8192 else 256

    def _get_tv_layout(self, num_copy_bits: int = 128) -> tuple[Any, Any]:
        vecsize = num_copy_bits // self.dtype.width
        assert self.N % vecsize == 0, f"N ({self.N}) must be divisible by vector size ({vecsize})"

        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0

        threads_per_row = self._calculate_threads_per_row()
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row)
        cols_per_block = num_threads // threads_per_row

        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    def _smem_size_in_bytes(self, tiler_mn: cute.Shape, num_warps: int) -> int:
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + num_warps * (cutlass.Float32.width // 8)
            + (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(self, mdY: cute.Tensor, mY: cute.Tensor, mdX: cute.Tensor) -> None:
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE

        self.kernel(mdY, mY, mdX, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mdY.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
        )

    @cute.kernel
    def kernel(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mdY.shape
        idX = cute.make_identity_tensor(shape)

        mdY = domain_offset_i64((bidx * tiler_mn[0], 0), mdY)
        mY = domain_offset_i64((bidx * tiler_mn[0], 0), mY)
        mdX = domain_offset_i64((bidx * tiler_mn[0], 0), mdX)

        gdY = cute.local_tile(mdY, tiler_mn, (0, 0))
        gY = cute.local_tile(mY, tiler_mn, (0, 0))
        gdX = cute.local_tile(mdX, tiler_mn, (0, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        sY = smem.allocate_tensor(
            mY.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        threads_per_row = tv_layout.shape[0][0]
        warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
        reduction_buffer = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_ordered_layout((num_warps // warps_per_row, warps_per_row), order=(1, 0)),
            byte_alignment=4,
        )

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mdY.element_type, num_bits_per_copy=128
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gdX.element_type, num_bits_per_copy=128
        )

        thr_copy_load = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn).get_slice(tidx)

        tdYgdY = thr_copy_load.partition_S(gdY)
        tdYsdY = thr_copy_load.partition_D(sdY)
        tYgY = thr_copy_load.partition_S(gY)
        tYsY = thr_copy_load.partition_D(sY)
        tdXgdX = thr_copy_store.partition_D(gdX)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        tdYrdY = cute.make_rmem_tensor_like(tdYgdY)
        tYrY = cute.make_rmem_tensor_like(tYgY)
        tdXrdX = cute.make_rmem_tensor_like(tdXgdX)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1])
        tdYpdY = (
            predicate_k(thr_copy_load.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tdYgdY, tdYsdY, pred=tdYpdY)
            cute.copy(copy_atom_load, tYgY, tYsY, pred=tdYpdY)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tdYsdY, tdYrdY)
        cute.autovec_copy(tYsY, tYrY)
        dy = tdYrdY.load().to(cute.Float32)
        y = tYrY.load().to(cute.Float32)

        dot = row_reduce(
            dy * y, cute.ReductionOp.ADD, threads_per_row, reduction_buffer, init_val=0.0
        )
        dx = y * (dy - dot)

        tdXrdX.store(dx.to(tdXrdX.element_type))

        tdXpdX = (
            predicate_k(thr_copy_store.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tdXrdX, tdXgdX, pred=tdXpdX)
