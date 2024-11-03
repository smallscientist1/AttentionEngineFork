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
# chunk_scan
####################################################################################################

def chunk_scan_triton(cb, x, dt, dA_cumsum, C, states):
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
    out, _ =  _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states)
    return out

def chunk_scan_ref(cb, x, dt, dA_cumsum, C, prev_states, D):
    from einops import rearrange, repeat
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, headdim = x.shape
    # _, _, ngroups, dstate = B.shape
    # assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    # assert C.shape == B.shape
    # B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)
    # CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
    #                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out

    return out

def chunk_scan_fwd(batch, seqlen, ngroups, nheads, headdim, dstate, block_M, block_N, block_K, block_Dstate):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504
    @T.prim_func
    def main(
        cb: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
        x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
        dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
        dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
        C: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
        prev_states: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype),
        D: T.Buffer((nheads), dtype),
        Output: T.Buffer((batch, seqlen, nheads, headdim), dtype)
    ):
        with T.Kernel(nheads, T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, threads=128) as (bz, bx, by):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.dyn")
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            # cb_shared_prev = T.alloc_shared((block_M, block_K), dtype)
            cb_local_prev = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_K), dtype, scope="shared")
            dA_cs_k_local = T.alloc_fragment((block_K), accum_dtype)
            # dA_cs_m_shared = T.alloc_shared((block_M), dtype, scope="shared")
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), dtype, scope="shared")
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.dyn")
            dA_cs_m_shared = T.alloc_shared((block_M), dtype, scope="shared")
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)
            D_local = T.alloc_fragment((1), accum_dtype)
            x_residual_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.dyn")
            x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)


            batch_idx = by % batch
            chunk_idx = by // batch
            # m: chunk_size
            # n : headdim
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            T.annotate_layout({
                acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared),
                cb_shared: tl.layout.make_swizzled_layout(cb_shared),
                x_residual_shared: tl.layout.make_swizzled_layout(x_residual_shared)
                # cb_shared_prev: tl.layout.make_swizzled_layout(cb_shared_prev)
            })
            
            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)
            
            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[batch_idx, 
                  chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                  bz // (nheads // ngroups),
                  0 : block_Dstate
                  ], 
                C_shared
            )
            T.copy(
                prev_states[batch_idx, 
                  chunk_idx,
                  bz,
                  n_idx * block_N : (n_idx + 1) * block_N,
                  0 : block_Dstate
                  ], 
                prev_state_shared
            )
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

            for k in T.Pipelined(loop_range, num_stages=2):
            # for k in T.Pipelined(
            #     loop_range, 
            #     num_stages=2, 
            #     order=[-1,0,-1,1,-1,2,-1,3,4], 
            #     stage=[-1,0,-1,0,-1,0,-1,0,0], 
            #     group=[[0],[1],[2],[3,4],[5],[6,7,8],[9],[10],[11]]
            # ):
                T.copy(
                    cb[batch_idx, 
                       chunk_idx, 
                       bz // (nheads // ngroups), 
                       m_idx * block_M : (m_idx + 1) * block_M, 
                       k * block_K : (k + 1) * block_K], 
                    cb_shared
                )
                T.copy(cb_shared, cb_local)
                T.copy(
                    dA_cumsum[batch_idx, 
                       bz, 
                       chunk_idx,
                       k * block_K : (k + 1) * block_K], 
                    dA_cs_k_shared
                )
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(
                        m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0
                    )
                T.copy(x[batch_idx, chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, bz, n_idx * block_N : (n_idx + 1) * block_N], x_shared)
                # T.copy(cb_local, cb_shared_prev)
                T.copy(cb_local, cb_local_prev)
                T.gemm(cb_local_prev, x_shared, acc_o)
            
            D_local[0] = D[bz]
            T.copy(x[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, bz, n_idx * block_N : (n_idx + 1) * block_N], x_residual_shared)
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]

            T.copy(acc_o, acc_o_shared)
            T.copy(acc_o_shared, Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, bz, n_idx * block_N : (n_idx + 1) * block_N])

    return main

def tune_chunk_scan_fwd(batch, seqlen, ngroups, nheads, headdim, dstate):
    
    def get_configs():
        block_M = [64, 128, 256]
        block_N = [32, 64]
        block_K = [64, 128, 256]
        block_Dstate = [128]
        num_stages = [1,2,3,4,5]
        # block_M = [64]
        # block_N = [64]
        # block_K = [64]
        # block_Dstate = [128]
        # num_stages = [2]
        _configs = list(itertools.product(block_M, block_N, block_K, block_Dstate, num_stages))

        configs = [
            {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'block_Dstate': c[3], 'num_stages': c[4], 'thread_num': c[0] * 2}
            for c in _configs
        ]
        return configs
    
    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'block_Dstate', 'num_stages', 'thread_num'], warmup=10, rep=5)
    @jit(out_idx=[6], supply_type=tl.TensorSupplyType.Normal, ref_prog=None, check_close=False, rtol=0.01, atol=0.01, profiler="tvm")
    def kernel(block_M = None, block_N = None, block_K = None, block_Dstate=None, num_stages = None, thread_num = None):
        dtype = "float16"
        accum_dtype = "float"
        nchunks = T.ceildiv(seqlen, chunk_size)
        p = 1.44269504
        @T.prim_func
        def main(
            cb: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
            x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
            dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
            dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
            C: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
            prev_states: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype),
            Output: T.Buffer((batch, seqlen, nheads, headdim), dtype)
        ):
            with T.Kernel(nheads, T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, threads=thread_num) as (bz, bx, by):
                acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
                acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
                cb_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.dyn")
                cb_local = T.alloc_fragment((block_M, block_K), dtype)
                # cb_shared_prev = T.alloc_shared((block_M, block_K), dtype)
                cb_local_prev = T.alloc_fragment((block_M, block_K), dtype)
                dA_cs_k_shared = T.alloc_shared((block_K), dtype, scope="shared")
                dA_cs_k_local = T.alloc_fragment((block_K), dtype)
                # dA_cs_m_shared = T.alloc_shared((block_M), dtype, scope="shared")
                dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
                dt_shared = T.alloc_shared((block_K), dtype, scope="shared")
                dt_local = T.alloc_fragment((block_K), accum_dtype)
                x_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.dyn")
                dA_cs_m_shared = T.alloc_shared((block_M), dtype, scope="shared")
                scale_m_local = T.alloc_fragment((block_M), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
                prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)


                batch_idx = by % batch
                chunk_idx = by // batch
                # m: chunk_size
                # n : headdim
                m_idx = bx // T.ceildiv(headdim, block_N)
                n_idx = bx % T.ceildiv(headdim, block_N)

                T.annotate_layout({
                    acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared),
                    cb_shared: tl.layout.make_swizzled_layout(cb_shared)
                    # cb_shared_prev: tl.layout.make_swizzled_layout(cb_shared_prev)
                })
                
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
                T.copy(dA_cs_m_shared, dA_cs_m_local)
                T.clear(acc_o)
                
                for i in T.Parallel(block_M):
                    scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
                T.copy(
                    C[batch_idx, 
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz // (nheads // ngroups),
                    0 : block_Dstate
                    ], 
                    C_shared
                )
                T.copy(
                    prev_states[batch_idx, 
                    chunk_idx,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                    0 : block_Dstate
                    ], 
                    prev_state_shared
                )
                T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    acc_o[i, j] *= scale_m_local[i]

                loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)
                # loop_range = T.ceildiv(chunk_size, block_K)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        cb[batch_idx, 
                        chunk_idx, 
                        bz // (nheads // ngroups), 
                        m_idx * block_M : (m_idx + 1) * block_M, 
                        k * block_K : (k + 1) * block_K], 
                        cb_shared
                    )
                    T.copy(cb_shared, cb_local)
                    T.copy(
                        dA_cumsum[batch_idx, 
                        bz, 
                        chunk_idx,
                        k * block_K : (k + 1) * block_K], 
                        dA_cs_k_shared
                    )
                    T.copy(dA_cs_k_shared, dA_cs_k_local)
                    for i, j in T.Parallel(block_M, block_K):
                        cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                    T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                    T.copy(dt_shared, dt_local)
                    for i, j in T.Parallel(block_M, block_K):
                        cb_local[i, j] *= dt_local[j]
                    for i, j in T.Parallel(block_M, block_K):
                        cb_local[i, j] = T.if_then_else(
                            m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0
                        )
                    T.copy(x[batch_idx, chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, bz, n_idx * block_N : (n_idx + 1) * block_N], x_shared)
                    # T.copy(cb_local, cb_shared_prev)
                    T.copy(cb_local, cb_local_prev)
                    T.gemm(cb_local_prev, x_shared, acc_o)
                T.copy(acc_o, acc_o_shared)
                T.copy(acc_o_shared, Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, bz, n_idx * block_N : (n_idx + 1) * block_N])

        return main
    return kernel()


if __name__ == "__main__":
    BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 8, 64, 1, 2048, 64, 128
    # BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 1, 1, 1, 64, 64, 128
    block_M, block_N, block_K, block_Dstate = 64, 64, 64, 128
    

    # chunk_scan_fwd
    total_flops = 2.0 * BATCH * SEQLEN * chunk_size * NHEADS * HEADDIM * 0.5 + 2.0 * BATCH * SEQLEN * NHEADS * HEADDIM * DSTATE
    # best_latency, best_config, ref_latency = tune_chunk_scan_fwd(BATCH, SEQLEN, NGROUPS, NHEADS, HEADDIM, DSTATE)
    # print(f"Best latency: {best_latency}")
    # print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    # print(f"Best config: {best_config}")
    # print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")
    program = chunk_scan_fwd(BATCH, SEQLEN, NGROUPS, NHEADS, HEADDIM, DSTATE, block_M, block_N, block_K, block_Dstate) 
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [7], tl.TensorSupplyType.Normal)
    mod.assert_allclose(chunk_scan_ref, rtol=0.1, atol=0.1)
    # latency = mod.do_bench(chunk_scan_triton, n_warmup=10, n_repeat=10, profiler="torch")
    # print("{:.4f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="auto")
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))