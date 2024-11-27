import torch
import numpy as np
import torch.nn.functional as F

import torch
from tvm import tl
import tvm.tl.language as T

from functools import partial

def convolution_transpose(N, C, H, W, F, K, S, D, P):
    KH, KW = K, K
    OH = (H - 1) * S - 2 * P + D * (KH - 1) + 1
    OW = (W - 1) * S - 2 * P + D * (KW - 1) + 1
    print("OH", OH, "OW", OW)

    dtype = "float16"
    accum_dtype = "float"
    block_M = 256
    block_N = 64
    block_K = 32

    @T.prim_func
    def main(
        data: T.Buffer((N, H, W, C), dtype),
        kernel: T.Buffer((KH, KW, C, F), dtype),
        out: T.Buffer((N, OH, OW, F), dtype),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=128) as (
            bx,
            by,
        ):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            kernel_flat = T.Buffer((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Buffer((N * OH * OW, F), dtype, out.data)

            T.annotate_layout({
                data_shared: tl.layout.make_swizzled_layout(data_shared),
                kernel_shared: tl.layout.make_swizzled_layout(kernel_shared),
            })

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=2):
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i

                    h_idx = ((m % (OH * OW)) // OW - k // (KW * C) * D + P)
                    w_idx = ((m % OW) - (k // C % KW) * D + P)
                    access_h = h_idx // S
                    access_w = w_idx // S

                    in_bound = ((h_idx % S == 0) and (w_idx % S == 0))
                    data_shared[i, j] = T.if_then_else(
                        in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0
                    )
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_flat[by * block_M, bx * block_N])

    return main

def convolution_transpose_fast(N, C, H, W, F, K, S, D, P):
    KH, KW = K, K
    OH = (H - 1) * S - 2 * P + D * (KH - 1) + 1
    OW = (W - 1) * S - 2 * P + D * (KW - 1) + 1
    print("OH", OH, "OW", OW)

    dtype = "float16"
    accum_dtype = "float"
    block_M = 128
    block_N = 128
    block_K = 64

    @T.prim_func
    def main(
        data: T.Buffer((N, H, W, C), dtype),
        kernel: T.Buffer((C, KH, KW, F), dtype),
        # out: T.Buffer((N, OH, OW, F), dtype),
        # TODO: modify output layout to nhwc
        out: T.Buffer((N, H, W, 2, 2, F), dtype),
    ):
        with T.Kernel(T.ceildiv(KH * KW * F, block_N), T.ceildiv(N * H * W, block_M), threads=128 * 2) as (
            bx,
            by,
        ):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            data_flat = T.Buffer((N * H * W, C), dtype, data.data)
            kernel_flat = T.Buffer((C, KH * KW * F), dtype, kernel.data)
            out_flat = T.Buffer((N * H * W, KH * KW * F), dtype, out.data)

            T.annotate_layout({
                data_shared: tl.layout.make_swizzled_layout(data_shared),
                kernel_shared: tl.layout.make_swizzled_layout(kernel_shared),
            })

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(C, block_K), num_stages=4):
                T.copy(data_flat[by * block_M, k_iter * block_K], data_shared)
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)
            T.copy(out_local, out_flat[by * block_M, bx * block_N])
    return main

def ref_program(A, B, stride, padding, dilation):
    A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
    B = B.permute(0, 3, 1, 2)  # C, KH, KW, F -> C, F, KH, KW
    C = torch.nn.functional.conv_transpose2d(A, B, stride=stride, padding=padding, dilation=dilation)
    C = C.permute(0, 2, 3, 1)  # N, F, OH, OW -> N, OH, OW, F
    N, OH, OW, F = C.shape
    C = C.view(N, OH // 2, 2, OW // 2, 2, F)
    C = C.permute(0, 1, 3, 2, 4, 5)
    return C


if __name__ == "__main__":
    # N, C, H, W, F, K, S, D, P = 32, 1024, 14, 14, 512, 2, 2, 1, 0
    N, C, H, W, F, K, S, D, P = 32, 256, 56, 56, 128, 2, 2, 1, 0
    program = convolution_transpose_fast(N, C, H, W, F, K, S, D, P)
    ref_program = partial(ref_program, stride=S, padding=P, dilation=D)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, atol=1e-1, rtol=1e-1)

    KH, KW = K, K
    OH = (H - 1) * S - 2 * P + D * (KH - 1) + 1
    OW = (W - 1) * S - 2 * P + D * (KW - 1) + 1
    total_flops = 2 * N * C * OH * OW * F * K * K

    latency = mod.do_bench(ref_program, n_warmup=10, n_repeat=10, profiler="torch")
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="torch")
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
