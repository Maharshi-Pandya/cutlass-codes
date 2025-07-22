import time
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch

from typing import Type
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def vectorized_relu_kernel(
    gIn: cute.Tensor,
    gOut: cute.Tensor,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    tid = bx * bdim + tx
    _, n = gIn.shape[1]
    mi = tid // n
    ni = tid % n

    a_val = gIn[(None, (mi, ni))].load()
    gOut[(None, (mi, ni))] = cute.where(a_val > 0, a_val, cute.full_like(a_val, 0))


@cute.jit
def vectorized_relu(
    mIn: cute.Tensor,
    mOut: cute.Tensor
):
    M, N = mIn.shape
    threads_per_block = 256

    gIn = cute.zipped_divide(mIn, (1, 4))
    gOut = cute.zipped_divide(mOut, (1, 4))

    vectorized_relu_kernel(gIn, gOut).launch(
        grid=(cute.size(gIn, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def run(
        M, N, dtype: Type[cutlass.Numeric],
        is_a_dynamic_layout = False,
        is_c_dynamic_layout = False,
        skip_ref_check = False,
        benchmark = True,
        warmup_iter = 10,
        iterations = 100
    ):
    print(f"\nRunning Relu test with:")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)
    a = torch.randn((M, N), dtype=torch_dtype, device="cuda")
    c = torch.zeros_like(a)

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    a_tensor = from_dlpack(a).mark_layout_dynamic() if is_a_dynamic_layout else from_dlpack(a)
    c_tensor = from_dlpack(c).mark_layout_dynamic() if is_c_dynamic_layout else from_dlpack(c)

    print("Compiling kernel with cute.compile ...")

    start_time = time.time()
    compiled_func = cute.compile(vectorized_relu, a_tensor, c_tensor)
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    print("Executing vectorized add kernel...")

    stream = cutlass_torch.current_stream()

    if not skip_ref_check:
        start = time.time()
        compiled_func(a_tensor, c_tensor)
        torch.cuda.synchronize()
        end = time.time()
        print(f">> Executed in {(end - start) * 1000} ms ...")
        print("Verifying results...")
        torch.testing.assert_close(torch.relu(a), c)
        print("Results verified successfully!")

    if not benchmark:
        return
    
    def generate_tensors():
        a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
        c = torch.zeros_like(a)

        a_tensor = from_dlpack(a).mark_layout_dynamic() if is_a_dynamic_layout else from_dlpack(a)
        c_tensor = from_dlpack(c).mark_layout_dynamic() if is_c_dynamic_layout else from_dlpack(c)
        return testing.JitArguments(a_tensor, c_tensor)
    
    avg_time_us = testing.benchmark(
        compiled_func, workspace_generator=generate_tensors,
        workspace_count=10,
        warmup_iterations=warmup_iter,
        profiling_iterations=iterations
    )

    print(f"Kernel execution time: {avg_time_us / 1e3:.4f} ms")
    print(
        f"Achieved memory throughput: {(2 * a.numel() * dtype.width // 8) / (avg_time_us / 1e6) / 1e9:.2f} GB/s"
    )


if __name__ == "__main__":
    run(8192, 8192, dtype=cutlass.Float32)