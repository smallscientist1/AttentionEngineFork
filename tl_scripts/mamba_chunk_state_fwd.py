import argparse
import torch
import torch.nn.functional as F
from tvm import tl
import tvm.tl.language as T
from tvm.tl.autotuner import *
from functools import partial
from einops import rearrange, repeat
import triton
import itertools

chunk_size = 256

####################################################################################################
# chunk_state
####################################################################################################

def chunk_state_triton(B, x, dt, dA_cumsum):
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd
    return _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=False)

def chunk_state_fwd(batch, seqlen, ngroups, nheads, headdim, dstate, block_M, block_N, block_K):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504
    @T.prim_func
    def main(
        B: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
        x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
        dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
        dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
        Output: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype)
    ):
        with T.Kernel(nheads, T.ceildiv(headdim, block_M) * T.ceildiv(dstate, block_N), batch * nchunks, threads=128) as (bz, bx, by):
            x_shared = T.alloc_shared((block_K, block_M), dtype)
            x_local = T.alloc_fragment((block_K, block_M), dtype)
            xt_local = T.alloc_fragment((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            dt_shared = T.alloc_shared((block_K), dtype)
            dA_cumsum_shared = T.alloc_shared((block_K), dtype)
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            scale = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_last = T.alloc_fragment((1), accum_dtype)
            dA_cumsum_local = T.alloc_fragment((block_K), accum_dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)

            loop_range = T.ceildiv(chunk_size, block_K)
            
            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(dstate, block_N)
            n_idx = bx % T.ceildiv(dstate, block_N)

            T.annotate_layout({
                x_shared: tl.layout.make_swizzled_layout(x_shared),
                acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)
            })
            
            dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx, chunk_size - 1]
            T.clear(acc_o)
            for k in T.Pipelined(loop_range, num_stages=4):
            # for k in T.Pipelined(
            #     loop_range, 
            #     num_stages=4, 
            #     order=[-1,-1,-1,1,-1,0],
            #     stage=[-1,-1,-1,0,-1,1],
            #     group=[[0],[1],[2],[3,4,5,6,7],[8],[9]],
            # ):
                T.copy(x[batch_idx, 
                    chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, 
                    bz, 
                    m_idx * block_M : (m_idx + 1) * block_M], 
                    x_shared)
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cumsum_shared)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dA_cumsum_shared, dA_cumsum_local)
                T.copy(dt_shared, dt_local)
                for i in T.Parallel(block_K):
                    scale[i] = T.exp2(dA_cs_last[0] * p - dA_cumsum_local[i] * p) * dt_local[i]
                T.copy(x_shared, x_local)
                for i, j in T.Parallel(block_M, block_K):
                    xt_local[i, j] = x_local[j, i] * scale[j]
                T.copy(B[batch_idx,
                    chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                    bz // (nheads // ngroups),
                    n_idx * block_N : (n_idx + 1) * block_N],
                    B_shared)
                T.gemm(xt_local, B_shared, acc_o)
            T.copy(acc_o, acc_o_shared)
            T.copy(acc_o_shared, Output[batch_idx, chunk_idx, bz, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N])
    return main


def tune_chunk_state_fwd(batch, seqlen, ngroups, nheads, headdim, dstate):

    def get_configs():
        block_M = [64, 128]
        block_N = [32, 64, 128]
        block_K = [32, 64]
        num_stages = [1, 2, 3, 4, 5]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages))

        configs = [
            {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'num_stages': c[3], 'thread_num': c[0] * 2}
            for c in _configs
        ]
        return configs
    
    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'num_stages', 'thread_num'], warmup=10, rep=5)
    @jit(out_idx=[4], supply_type=tl.TensorSupplyType.Normal, ref_prog=None, rtol=0.01, atol=0.01, profiler="torch")
    def kernel(block_M = None, block_N = None, block_K = None, num_stages = None, thread_num = None):
        dtype = "float16"
        accum_dtype = "float"
        nchunks = T.ceildiv(seqlen, chunk_size)
        p = 1.44269504
        @T.prim_func
        def main(
            B: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
            x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
            dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
            dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
            Output: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype)
        ):
            with T.Kernel(T.ceildiv(headdim, block_M) * T.ceildiv(dstate, block_N), batch * nchunks, nheads, threads=thread_num) as (bx, by, bz):
                x_shared = T.alloc_shared((block_K, block_M), dtype)
                x_local = T.alloc_fragment((block_K, block_M), dtype)
                xt_local = T.alloc_fragment((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                dt_shared = T.alloc_shared((block_K), dtype)
                dA_cumsum_shared = T.alloc_shared((block_K), dtype)
                acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
                acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
                scale = T.alloc_fragment((block_K), accum_dtype)
                dA_cs_last = T.alloc_fragment((1), accum_dtype)
                dA_cumsum_local = T.alloc_fragment((block_K), accum_dtype)
                dt_local = T.alloc_fragment((block_K), accum_dtype)

                loop_range = T.ceildiv(chunk_size, block_K)
                
                batch_idx = by % batch
                chunk_idx = by // batch
                m_idx = bx // T.ceildiv(dstate, block_N)
                n_idx = bx % T.ceildiv(dstate, block_N)

                T.annotate_layout({
                    acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)
                })
                
                dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx, chunk_size - 1]
                T.clear(acc_o)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(x[batch_idx, 
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, 
                        bz, 
                        m_idx * block_M : (m_idx + 1) * block_M], 
                        x_shared)
                    T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cumsum_shared)
                    T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                    T.copy(dA_cumsum_shared, dA_cumsum_local)
                    T.copy(dt_shared, dt_local)
                    for i in T.Parallel(block_K):
                        scale[i] = T.exp2(dA_cs_last[0] * p - dA_cumsum_local[i] * p) * dt_local[i]
                    T.copy(x_shared, x_local)
                    for i, j in T.Parallel(block_M, block_K):
                        xt_local[i, j] = x_local[j, i] * scale[j]
                    T.copy(B[batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz // (nheads // ngroups),
                        n_idx * block_N : (n_idx + 1) * block_N],
                        B_shared)
                    T.gemm(xt_local, B_shared, acc_o)
                T.copy(acc_o, acc_o_shared)
                T.copy(acc_o_shared, Output[batch_idx, chunk_idx, bz, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N])
        return main
    return kernel()

if __name__ == "__main__":
    BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 1, 64, 1, 1024, 64, 128
    # BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 1, 1, 1, 64, 64, 128
    block_M, block_N, block_K = 64, 128, 64
    

    # chunk_state_fwd
    total_flops = 2 * BATCH * SEQLEN * NHEADS * HEADDIM * DSTATE
    # best_latency, best_config, ref_latency = tune_chunk_state_fwd(BATCH, SEQLEN, NGROUPS, NHEADS, HEADDIM, DSTATE)
    # print(f"Best latency: {best_latency}")
    # print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    # print(f"Best config: {best_config}")
    # print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")
    program = chunk_state_fwd(BATCH, SEQLEN, NGROUPS, NHEADS, HEADDIM, DSTATE, block_M, block_N, block_K) 
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [4], tl.TensorSupplyType.Normal)
    # mod.assert_allclose(chunk_state_triton, rtol=0.1, atol=0.1)
    latency = mod.do_bench(chunk_state_triton, n_warmup=10, n_repeat=10, profiler="torch")
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="auto")
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))