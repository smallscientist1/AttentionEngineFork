import torch
from tvm import tl
import tvm.tl.language as T
from tvm.tl.autotuner import *
import itertools
import argparse
from functools import partial

def get_configs():
    block_M = [64,128,256]
    block_N = [64,128,256]
    block_K = [32,64]
    num_stages = [1,2,3,4]
    thread_num = [128,256]
    # block_M = [128]
    # block_N = [128]
    # block_K = [64]
    # num_stages = [4]
    # thread_num = [128]
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages, thread_num))

    configs = [
        {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'num_stages': c[3], 'thread_num': c[4]}
        for c in _configs
    ]
    return configs

def convolution_transpose_fast(N, C, H, W, F, K, S, D, P):
    KH, KW = K, K
    OH = (H - 1) * S - 2 * P + D * (KH - 1) + 1
    OW = (W - 1) * S - 2 * P + D * (KW - 1) + 1

    dtype = "float16"
    accum_dtype = "float"

    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'num_stages', 'thread_num'], warmup=10, rep=10)
    @jit(out_idx=[2], supply_type=tl.TensorSupplyType.Integer, ref_prog=None, profiler="auto")
    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None):
        @T.prim_func
        def main(
            data: T.Buffer((N, H, W, C), dtype),
            kernel: T.Buffer((C, KH, KW, F), dtype),
            # out: T.Buffer((N, OH, OW, F), dtype),
            # TODO: modify output layout to nhwc
            out: T.Buffer((N, H, W, 2, 2, F), dtype),
        ):
            with T.Kernel(T.ceildiv(KH * KW * F, block_N), T.ceildiv(N * H * W, block_M), threads=thread_num) as (
                bx,
                by,
            ):
                data_shared = T.alloc_shared((block_M, block_K), dtype)
                kernel_shared = T.alloc_shared((block_K, block_N), dtype)
                out_shared = T.alloc_shared((block_M, block_N), dtype)
                out_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                data_flat = T.Buffer((N * H * W, C), dtype, data.data)
                kernel_flat = T.Buffer((C, KH * KW * F), dtype, kernel.data)
                out_flat = T.Buffer((N * H * W, KH * KW * F), dtype, out.data)

                T.annotate_layout({
                    data_shared: tl.layout.make_swizzled_layout(data_shared),
                    kernel_shared: tl.layout.make_swizzled_layout(kernel_shared),
                    out_shared: tl.layout.make_swizzled_layout(out_shared),
                })

                T.clear(out_local)
                for k_iter in T.Pipelined(T.ceildiv(C, block_K), num_stages=num_stages):
                    T.copy(data_flat[by * block_M, k_iter * block_K], data_shared)
                    T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                    T.gemm(data_shared, kernel_shared, out_local)
                T.copy(out_local, out_shared)
                T.copy(out_shared, out_flat[by * block_M, bx * block_N])
        return main
    return kernel()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=128, help='n')
    parser.add_argument('--c', type=int, default=128, help='c')
    parser.add_argument('--h', type=int, default=64, help='h')
    parser.add_argument('--w', type=int, default=64, help='w')
    parser.add_argument('--f', type=int, default=128, help='f')
    parser.add_argument('--k', type=int, default=3, help='k')
    parser.add_argument('--s', type=int, default=1, help='s')
    parser.add_argument('--d', type=int, default=1, help='d')
    parser.add_argument('--p', type=int, default=1, help='p')
    args = parser.parse_args()
    N, C, H, W, F, K, S, D, P = args.n, args.c, args.h, args.w, args.f, args.k, args.s, args.d, args.p
    assert K == 2 and S == 2 and D == 1 and P == 0, "Not supported"
    KH, KW = K, K
    OH = (H - 1) * S - 2 * P + D * (KH - 1) + 1
    OW = (W - 1) * S - 2 * P + D * (KW - 1) + 1
    total_flops = 2 * N * C * OH * OW * F * K * K
    best_latency, best_config, ref_latency = convolution_transpose_fast(N, C, H, W, F, K, S, D, P)
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")
    print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")