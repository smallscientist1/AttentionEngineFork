import torch
from tvm import tl
import tvm.tl.language as T

from functools import partial


def convolution(N, C, H, W, F, K, S, D, P):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

    dtype = "float16"
    accum_dtype = "float"
    block_M = 128
    block_N = 256
    block_K = 64

    @T.prim_func
    def main(
        data: T.Buffer((N, H, W, C), dtype),
        kernel: T.Buffer((KH, KW, C, F), dtype),
        out: T.Buffer((N, OH, OW, F), dtype),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=128 * 2) as (
            bx,
            by,
        ):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Buffer((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Buffer((N * OH * OW, F), dtype, out.data)

            T.annotate_layout({
                out_shared: tl.layout.make_swizzled_layout(out_shared),
                data_shared: tl.layout.make_swizzled_layout(data_shared),
                kernel_shared: tl.layout.make_swizzled_layout(kernel_shared),
            })

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=4):
                T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                # for i, j in T.Parallel(block_M, block_K):
                #     k = k_iter * block_K + j
                #     m = by * block_M + i
                #     access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                #     access_w = m % OW * S + k // C % KW * D - P
                #     in_bound = (
                #         (access_h >= 0)
                #         and (access_w >= 0)
                #         and (access_h < H)
                #         and (access_w < W)
                #     )
                #     data_shared[i, j] = T.if_then_else(
                #         in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0
                #     )
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
    # N, C, H, W, F, K, S, D, P = 8, 256, 28, 28, 256, 3, 2, 1, 1
    # N, C, H, W, F, K, S, D, P = 128, 3, 224, 224, 64, 7, 2, 1, 3
    N, C, H, W, F, K, S, D, P = 128, 256, 56, 56, 512, 1, 2, 1, 0

    program = convolution(N, C, H, W, F, K, S, D, P)
    ref_program = partial(ref_program, stride=S, padding=P, dilation=D)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)

    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    total_flops = 2 * N * C * OH * OW * F * K * K

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="auto")
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
