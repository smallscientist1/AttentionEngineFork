import torch
from tvm import tl
import tvm.tl.language as T

from functools import partial


def convolution(N, C, H, W, INC, KW, KH, P, S, D):
    INH = (H - 1) * S + (KH - 1) * D + 1 - 2 * P
    INW = (W - 1) * S + (KW - 1) * D + 1 - 2 * P

    dtype = "float16"
    accum_dtype = "float"
    block_M = 128
    block_N = 256
    block_K = 64

    @T.prim_func
    def main(
        data: T.Buffer((N, INH, INW, INC), dtype),
        kernel: T.Buffer((KH, KW, INC, C), dtype),
        out: T.Buffer((N, H, W, C), dtype),
    ):
        with T.Kernel(T.ceildiv(C, block_N), T.ceildiv(N * H * W, block_M), threads=128 * 2) as (
            bx,
            by,
        ):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Buffer((KH * KW * INC, C), dtype, kernel.data)
            out_flat = T.Buffer((N * H * W, C), dtype, out.data)

            T.annotate_layout({
                out_shared: tl.layout.make_swizzled_layout(out_shared),
                data_shared: tl.layout.make_swizzled_layout(data_shared),
                kernel_shared: tl.layout.make_swizzled_layout(kernel_shared),
            })

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * INC, block_K), num_stages=4):
                T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return main


def ref_program(A, B, stride, padding, dilation):
    A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
    B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
    C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
    C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
    return C


if __name__ == "__main__":
    # N, C, H, W, INC, KW, KH, P, S, D = 128, 128, 28, 28, 128, 3, 3, 1, 1, 1
    N, C, H, W, INC, KW, KH, P, S, D = 128, 256, 28, 28, 256, 3, 3, 1, 2, 1
    program = convolution(N, C, H, W, INC, KW, KH, P, S, D)
    ref_program = partial(ref_program, stride=S, padding=P, dilation=D)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)

    total_flops = 2 * N * C * H * W * INC * KH * KW

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="auto")
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
