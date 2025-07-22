import torch
import lovely_tensors as lt; lt.monkey_patch()

import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack


@cute.kernel
def kernel(
    gIn: cute.Tensor,
    gOut: cute.Tensor,
    tv_layout: cute.Layout,
    tiler_mn: cute.Tiler
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()

    # ((TileM, TileN), (RestM, RestN))
    iters = gIn.shape[1][1]
    print("gIn layout:", gIn.layout)

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gIn.element_type)
    tiled_copy = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)

    # TODO: find a better way to get target rmem layout
    thr_copy_slice = tiled_copy.get_slice(tx)
    target_layout = thr_copy_slice.partition_S(gIn[((None, None), 0)]).layout 
    frgIn = cute.make_fragment(target_layout, dtype=gIn.element_type)

    print(f"Fragment layout: {frgIn.layout}\n")

    # one block -> one row
    # a row is divided into tiles
    # threads perform vectorized loads
    for i in cutlass.range_constexpr(iters):
        bIn = gIn[((None, None), (bx, i))]
        bOut = gOut[((None, None), (bx, i))]

        tIn = thr_copy_slice.partition_S(bIn)
        tOut = thr_copy_slice.partition_D(bOut)
        cute.copy(copy_atom_load, tIn, frgIn)
        cute.copy(copy_atom_load, frgIn, tOut)


@cute.jit
def host(
    x: cute.Tensor,
    y: cute.Tensor,
    copy_bits: cutlass.Constexpr = 128
):
    dtype = x.element_type
    vecsize = copy_bits // dtype.width

    thr_layout = cute.make_layout((1, 256))
    val_layout = cute.make_layout((1, vecsize), stride=(0, 1))

    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gIn = cute.zipped_divide(x, tiler_mn)
    gOut = cute.zipped_divide(y, tiler_mn)

    print("Tiler (per block): ", tiler_mn)
    print(f"TV layout: {tv_layout}\n")

    kernel(gIn, gOut, tv_layout, tiler_mn).launch(
        grid=(x.shape[0], 1, 1),
        block=(256, 1, 1)
    )


x = torch.ones((2048, 9001), device="cuda", dtype=torch.bfloat16)
x_tensor = from_dlpack(x)

y = torch.zeros_like(x)
y_tensor = from_dlpack(y)

compiled = cute.compile(host, x_tensor, y_tensor)
compiled(x_tensor, y_tensor)

print("Input tensor: ", x)
print("Output tensor: ", y)

torch.testing.assert_close(x, y)
