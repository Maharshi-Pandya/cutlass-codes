"""
elementwise binary operation using cutlass/cute DSL

showcases:
- global memory to register copies
- tiled copying
- partioning from source data
- TV (Thread-Value) layout mapping

given 2D input tensors A and B, output tensor C is computed as:

Cij = op(Aij, Bij)

where `op` is a elementwise binary operation e.g. +, -, *, / and so on...

all tensors will be row major, leading dimension of 1 in the right most dimension
"""

import time
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch

import cuda.bindings.driver as cuda
from typing import Type
from cutlass.cute.runtime import from_dlpack


class ElementwiseOp:
    def __init__(self, n_threads_per_cta: int = 256):
        self.n_threads_per_cta = n_threads_per_cta
    
    @cute.kernel
    def kernel(
        self, gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
        gCoord: cute.Tensor, shape: cute.Shape, 
        tv_layout: cute.Layout, tiler_mn: cute.Tiler
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()

        # CTA slice after zipped divide
        # shape of b* will be (TileM, TileN) after zipped divide with TV layout tiler
        # for CTA level slice we use second mode to access tile of a block idx
        coord = ((None, None), bx)
        bA = gA[coord]
        bB = gB[coord]
        bC = gC[coord]
        bCoord = gCoord[coord]

        # declare a copy atom, op that will be used for memory copy
        copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
        copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

        # define tiled copy CTA; contains info about how to copy elements for all threads
        tiled_copy_A = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        tiled_copy_B = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
        tiled_copy_C = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        # get THIS thread's copy slice
        tA_copy_slice = tiled_copy_A.get_slice(tx)
        tB_copy_slice = tiled_copy_B.get_slice(tx)
        tC_copy_slice = tiled_copy_C.get_slice(tx)

        # partition the thread's slice from the CTA as source for copy
        # _S means source _D means destination (good to bifurcate)
        tA = tA_copy_slice.partition_S(bA)
        tB = tB_copy_slice.partition_S(bB)
        tC = tC_copy_slice.partition_D(bC)

        # allocate fragments for gmem -> rmem copy
        tAfA = cute.make_fragment_like(tA)
        tBfB = cute.make_fragment_like(tB)
        tCfC = cute.make_fragment_like(tC)

        # predicate copy for OOB reads
        tCoord = tC_copy_slice.partition_S(bCoord)
        fPred = cute.make_fragment(tCoord.shape, cutlass.Boolean)

        for i in cutlass.range_constexpr(0, cute.size(tCoord), 1):
            # is current index less than shape?
            # only values with true will be copied when
            # shape is not divisible by tile size
            val = cute.elem_less(tCoord[i], shape)
            fPred[i] = val

        # move data now
        cute.copy(copy_atom_load, tA, tAfA, pred=fPred)
        cute.copy(copy_atom_load, tB, tBfB, pred=fPred)

        # perform operation
        res = tAfA.load() + tBfB.load()
        tCfC.store(res)

        # store the result back
        cute.copy(copy_atom_store, tCfC, tC, pred=fPred)

    @cute.jit
    def host(self, mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
        dtype = mA.element_type
        vecsize = copy_bits // dtype.width

        # the order parameter specifies the ordering of dimensions from fastest-varying to slowest-varying
        # for a 2D tensor, (0,1) creates a column-major layout, while (1,0) creates a row-major layout
        # the length of order must match the rank of the shape
        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((4, vecsize), order=(1, 0))

        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gA = cute.zipped_divide(mA, tiler_mn)
        gB = cute.zipped_divide(mB, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        # an identity tensor maps each coordinate to itself, effectively creating a 
        # counting sequence within the shape's bounds. 
        # this is useful for generating coordinate indices or creating reference tensors 
        # for layout transformations.
        idC = cute.make_identity_tensor(mC.shape)
        gCoord = cute.zipped_divide(idC, tiler_mn)

        self.kernel(gA, gB, gC, gCoord, mC.shape, tv_layout, tiler_mn).launch(
            grid=[cute.size(gC, mode=[1]), 1, 1],
            # good way to find number of threads
            block=[cute.size(tv_layout, mode=[0]), 1, 1]
        )


def run(
        M, N, dtype: Type[cutlass.Numeric],
        is_a_dynamic_layout = False,
        is_b_dynamic_layout = False,
        is_c_dynamic_layout = False,
        skip_ref_check = False,
        benchmark = True,
        warmup_iter = 10,
        iterations = 100
    ):
    print(f"\nRunning Elementwise Add test with:")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)
    a = torch.randn((M, N), dtype=torch_dtype, device="cuda")
    b = torch.randn((M, N), dtype=torch_dtype, device="cuda")

    c = torch.zeros_like(a)

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    a_tensor = from_dlpack(a).mark_layout_dynamic() if is_a_dynamic_layout else a
    b_tensor = from_dlpack(b).mark_layout_dynamic() if is_b_dynamic_layout else b
    c_tensor = from_dlpack(c).mark_layout_dynamic() if is_c_dynamic_layout else c

    print("Compiling kernel with cute.compile ...")
    driver = ElementwiseOp()
    start_time = time.time()
    compiled_func = cute.compile(driver.host, a_tensor, b_tensor, c_tensor)
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    print("Executing vectorized add kernel...")

    stream = cutlass_torch.current_stream()

    if not skip_ref_check:
        compiled_func(a_tensor, b_tensor, c_tensor)
        print("Verifying results...")
        torch.testing.assert_close(a + b, c)
        print("Results verified successfully!")

    if not benchmark:
        return
    
    def generate_tensors():
        a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
        c = torch.zeros_like(a)

        a_tensor = from_dlpack(a).mark_layout_dynamic() if is_a_dynamic_layout else a
        b_tensor = from_dlpack(b).mark_layout_dynamic() if is_b_dynamic_layout else b
        c_tensor = from_dlpack(c).mark_layout_dynamic() if is_c_dynamic_layout else c
        return testing.JitArguments(a_tensor, b_tensor, c_tensor)
    
    avg_time_us = testing.benchmark(
        compiled_func, workspace_generator=generate_tensors,
        workspace_count=10,
        warmup_iterations=warmup_iter,
        profiling_iterations=iterations
    )

    print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
    print(
        f"Achieved memory throughput: {(3 * a.numel() * dtype.width // 8) / (avg_time_us / 1e6) / 1e9:.2f} GB/s"
    )


if __name__ == "__main__":
    M, N = 2048, 2048

    run(
        M,
        N,
        dtype=cutlass.BFloat16,
        is_a_dynamic_layout=True,
        is_b_dynamic_layout=True,
        is_c_dynamic_layout=True,
        skip_ref_check=False,
        benchmark=True,
        warmup_iter=10,
        iterations=100,
    )
