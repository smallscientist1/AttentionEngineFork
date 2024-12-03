import torch
from tvm import tl
import tvm.tl.language as T
from functools import partial
from tvm.tl.autotuner import *
import itertools
import argparse

# Codegen bug:
#   LoadK should wait for MMA0 done
# @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
# def tvm_callback_cuda_postproc(code, _):
#     code = code.replace("""tl::mbarrier_wait(_mbarrier[1], ((k & 1) ^ 1));""", 
# """tl::mbarrier_wait(_mbarrier[1], ((k & 1))); // replace""")
#     code = code.replace("""tl::gemm_ss<64, 64, 64, 4, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(acc_s[0])));
#     #pragma unroll""", 
# """tl::gemm_ss<64, 64, 64, 4, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(acc_s[0])));
#     tl::mbarrier_arrive(_mbarrier[1]);
#     #pragma unroll // replace""")
#     return code

# loadk(0)
# gemm0(0)
# loadk(1)
# softmax(0)
# loadv(0)

# for i in range(loop_range - 2):
#   gemm0(i+1)
#   gemm1(i+0)
#   loadk(i+2)
#   softmax(i+1)
#   loadv(i+1)

# gemm0(loop_range - 1)
# gemm1(loop_range - 2)
# softmax(loop_range - 1)
# loadv(loop_range - 1)
# gemm1(loop_range - 1)

def ref_program(Q, K, V, casual):
    # from flash_attn.flash_attn_interface import flash_attn_func
    import sys
    sys.path.append("/home/msra/cy/tvm.tl/3rdparty/fa3")
    from hopper.flash_attn_interface import flash_attn_func
    ret = flash_attn_func(Q, K, V, causal=casual)
    return ret[0]

def get_configs():
    block_M = [64, 128]
    block_N = [64 + i * 16 for i in range(13)]
    # block_N = [80]
    num_stages = [1, 2, 3, 4]

    _configs = list(itertools.product(block_M, block_N, num_stages))

    configs = [
        {'block_M': c[0], 'block_N': c[1], 'num_stages': c[2], 'thread_num': c[0] * 2}
        for c in _configs
    ]
    return configs

def flashattn(batch, heads, seq_len, dim, is_casual):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'num_stages', 'thread_num'], warmup=10, rep=5)
    @jit(out_idx=[3], supply_type=tl.TensorSupplyType.Normal, ref_prog=None, rtol=0.01, atol=0.01, profiler="auto")
    def kernel(block_M = None, block_N = None, num_stages = None, thread_num = None):
        @T.macro
        def MMA0(
            K: T.Buffer(shape, dtype),
            Q_shared: T.Buffer([block_M, dim], dtype),
            K_shared: T.Buffer([block_N, dim], dtype),
            acc_s: T.Buffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
            if is_casual:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                    )
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Buffer(shape, dtype),
            V_shared: T.Buffer([block_M, dim], dtype),
            acc_s_cast: T.Buffer([block_M, block_N], dtype),
            acc_o: T.Buffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
            acc_s: T.Buffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.Buffer([block_M, block_N], dtype),
            scores_max: T.Buffer([block_M], accum_dtype),
            scores_max_prev: T.Buffer([block_M], accum_dtype),
            scores_scale: T.Buffer([block_M], accum_dtype),
            scores_sum: T.Buffer([block_M], accum_dtype),
            logsum: T.Buffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
            acc_o: T.Buffer([block_M, dim], accum_dtype),
            scores_scale: T.Buffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def main(
            Q: T.Buffer(shape, dtype),
            K: T.Buffer(shape, dtype),
            V: T.Buffer(shape, dtype),
            Output: T.Buffer(shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # loop_range = (
                #     T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
                # )
                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)) if is_casual else T.ceildiv(seq_len, block_N)
                )

                # for k in T.Pipelined(loop_range, num_stages=num_stages, order=[-1,0,3,1,-1,2], stage=[-1,0,0,1,-1,1], sync=[[0,13],[1,9]], group=[[0], [1,2], [3,4,5,6,7,8,9,10], [11], [12], [13]]):
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

        return main
    return kernel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, default=1, help='batch')
    parser.add_argument('--h', type=int, default=16, help='num_heads')
    parser.add_argument('--seq', type=int, default=8192, help='seqlen')
    parser.add_argument('--d', type=int, default=64, help='head_dim')
    parser.add_argument('--casual', action='store_true', help='casual')

    args = parser.parse_args()

    BATCH, H, N_CTX, D_HEAD, casual = args.b, args.h, args.seq, args.d, args.casual

    print(f"Tuning for batch={BATCH}, heads={H}, seq_len={N_CTX}, dim={D_HEAD}, casual={casual}")
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    best_latency, best_config, ref_latency = flashattn(BATCH, H, N_CTX, D_HEAD, casual)
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")
    # print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")