# fwd 1,32,4096,128
# 135 tflops
# no score_mod(0.5 only) : 545 TFlops
# --parallel-for--> 134 tflops
# --tanh+bias only--> 545 tflops
# --tanh+1+bias--> 545 tflops
# --tanh+1+bias+0.5--> 545 tflops

# cuda bug: not execute tanh

# fast_tanh: 317 tflops --applynotunrollmask--> 315 tflops  --parallel-for-unroll--> 463 tflops --correct softmax_bias pose-->: 444 tfloops


# bwd
# 
# tl bug： TMA outside pipeline cause stage false
# 1,32,2048,128
# fast_tanh: 312 tflops --pipeline--> 264 tflops--sync--> 264 tflops --nocausalmask-->285tflops
# (not actual pipeline)

import torch
from tvm import tl
import tvm.tl.language as T

import itertools

def fast_tanh(A, B):
    return T.call_extern("handle", "fasttanh", T.address_of(A), T.address_of(B))

def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(
        dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2]
    )

def kernel(batch, heads, seq_len, dim, dimv, 
        block_M = None, block_N = None, num_stages = None, thread_num = None):
    # scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e) # 0.69314718  loge(2)
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    dtype = "float16"
    accum_dtype = "float"
    
    # TODO: mask
    is_casual = True


    @T.macro
    def score_mod(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        scores: T.Buffer([block_M, block_N], accum_dtype), 
        softmax_bias: T.Buffer([1], accum_dtype), 

        ):
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = scores[i0,i1] + softmax_bias[0]
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = scores[i0,i1] * 0.5
        for i0,i1 in T.Parallel(block_M,block_N):
            # scores[i0,i1] = T.tanh(scores[i0,i1])
            fast_tanh(scores[i0,i1], scores[i0,i1])
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = scores[i0,i1] + 1
        for i0,i1 in T.Parallel(block_M,block_N):
            scores[i0,i1] = scores[i0,i1] * 0.5

    
    @T.macro
    def online_func(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        scores: T.Buffer([block_M, block_N], accum_dtype), 
        o_scale: T.Buffer([block_M], accum_dtype), 

    ):
        pass

        
    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape, dtype), # type: ignore
        V: T.Buffer(shape_v, dtype), # type: ignore
        g_softmax_bias: T.Buffer([1], accum_dtype), 


        Output: T.Buffer(shape_v, dtype), # type: ignore
        
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_local = T.alloc_fragment([block_N, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dimv], dtype)
            V_local = T.alloc_fragment([block_N, dimv], dtype)
            # acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            # acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

            softmax_bias = T.alloc_fragment([1], accum_dtype)

            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            o_scale = T.alloc_fragment([block_M], accum_dtype)
            acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

            

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            softmax_bias[0] = g_softmax_bias[0]

            T.fill(acc_o, 0)
            T.fill(o_scale, 1)

            

            # TODO: mask
            loop_range = (
                T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )

            loop_range_fullblock = (
                T.floordiv((bx) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )
            loop_range_maskblock = (
                T.floordiv((bx) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N),
                T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )


            for k in T.Pipelined(loop_range, num_stages=num_stages,
                                #  order=[-1,0,-1,3,1,2], 
                                #  stage=[-1,0,-1,0,1,1],
                                #  group=[[0],[1,2],[3],[4,5,6,7,8,9],[10],[11]]
                                 ):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                # TODO: copy custom_fwd_input_tensor in score_mod&online_func
                # ...

                # TODO: naive solution: if reduce_max, -T.inf; if reduce_sum, 0
                if is_casual:# and False:
                    for i, j in T.Parallel(block_M, block_N):
                        scores[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j, 0, -T.infinity(scores.dtype)
                        )
                else:
                    T.clear(scores)
                
                T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                    
                # call score_mod
                score_mod(scores, softmax_bias) # scores
                    
                # call online_func
                online_func(scores, o_scale) # scores
                
                T.copy(scores, acc_s_cast)
                # for i, j in T.Parallel(block_M, dimv):
                #     acc_o[i, j] *= o_scale[i]
                for i, j in T.Parallel(block_M, dimv):
                    acc_o[i, j] *= o_scale[i]

                
                # update online_rowscales
                

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            
            # mask loop
            # with T.floordiv(bx * block_M, block_N) as k1
            # if loop_range_maskblock[0] == loop_range_maskblock[1] - 1:
            #     k1 =  loop_range_maskblock[0]
            #     T.copy(K[bz, k1 * block_N : (k1 + 1) * block_N, by, :], K_local)
            #     T.copy(K_local,K_shared)
            #     # T.copy(K[bz, k1 * block_N : (k1 + 1) * block_N, by, :], K_shared)

            #     # TODO: copy custom_fwd_input_tensor in score_mod&online_func
            #     # ...

            #     # TODO: naive solution: if reduce_max, -T.inf; if reduce_sum, 0
            #     if is_casual:# and False:
            #         for i, j in T.Parallel(block_M, block_N):
            #             scores[i, j] = T.if_then_else(
            #                 bx * block_M + i >= k1 * block_N + j, 0, -T.infinity(scores.dtype)
            #             )
            #     else:
            #         T.clear(scores)
                
            #     T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            #     # T.copy(V[bz, k1 * block_N : (k1 + 1) * block_N, by, :], V_shared)
            #     T.copy(V[bz, k1 * block_N : (k1 + 1) * block_N, by, :], V_local)
            #     T.copy(V_local, V_shared)
                    
            #     # call score_mod
            #     score_mod(scores, softmax_bias) # scores
                    
            #     # call online_func
            #     online_func(scores, o_scale) # scores

            #     # for i, j in T.Parallel(block_M, dimv):
            #     #     acc_o[i, j] *= o_scale[i]
            #     for i, j in T.Parallel(block_M, dimv):
            #         acc_o[i, j] *= o_scale[i]

                
            #     # update online_rowscales
                

            #     T.copy(scores, acc_s_cast)
            #     T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
    
            # online_fwd_epilogue
            

            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

            # save final_rowscale
            
        
    return main



def flashattn_bwd_preprocess(batch, heads, seq_len, dim, dimv):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Buffer(shape, dtype), # type: ignore
        dO: T.Buffer(shape, dtype), # type: ignore
        Delta: T.Buffer([batch, heads, seq_len], accum_dtype), # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dimv, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep

def flashattn_bwd(batch, heads, seq_len, dim, dimv, is_casual, 
                block_M, block_N, thread_num = 128):
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    dtype = "float16"
    accum_dtype = "float"

    @T.macro
    def score_mod(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        qkT: T.Buffer([block_M, block_N], accum_dtype), 
        qkT_0_0_0: T.Buffer([block_M, block_N], accum_dtype), 
        softmax_bias: T.Buffer([1], accum_dtype), 

        ):
        for i0,i1 in T.Parallel(block_M,block_N):
            qkT[i0,i1] = qkT[i0,i1] + softmax_bias[0]
        for i0,i1 in T.Parallel(block_M,block_N):
            qkT[i0,i1] = qkT[i0,i1] * 0.5
        for i0,i1 in T.Parallel(block_M,block_N):
            # qkT[i0,i1] = T.tanh(qkT[i0,i1])
            fast_tanh(qkT[i0,i1], qkT[i0,i1])
        for i0,i1 in T.Parallel(block_M,block_N):
            qkT_0_0_0[i0,i1] = qkT[i0,i1] + 1
        for i0,i1 in T.Parallel(block_M,block_N):
            qkT_0_0_0[i0,i1] = qkT_0_0_0[i0,i1] * 0.5


    
    @T.macro
    def score_mod_backward(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        dsT: T.Buffer([block_M, block_N], accum_dtype), 
        dsT_0_0: T.Buffer([block_M, block_N], accum_dtype), 
        qkT: T.Buffer([block_M, block_N], accum_dtype), 
        qkT_0_0_0: T.Buffer([block_M, block_N], accum_dtype), 
        softmax_bias: T.Buffer([1], accum_dtype), 

    ):
        for i0,i1 in T.Parallel(block_M,block_N):
            dsT[i0,i1] = dsT[i0,i1] * 0.5
        for i0,i1 in T.Parallel(block_M,block_N):
            dsT_0_0[i0,i1] = dsT[i0,i1] * qkT[i0,i1]
        for i0,i1 in T.Parallel(block_M,block_N):
            dsT_0_0[i0,i1] = dsT_0_0[i0,i1] * qkT[i0,i1]
        for i0,i1 in T.Parallel(block_M,block_N):
            dsT[i0,i1] = dsT[i0,i1] - dsT_0_0[i0,i1]
        for i0,i1 in T.Parallel(block_M,block_N):
            dsT[i0,i1] = dsT[i0,i1] * 0.5


    @T.prim_func
    def flash_bwd(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape, dtype), # type: ignore
        V: T.Buffer(shape, dtype), # type: ignore
        dO: T.Buffer(shape, dtype), # type: ignore

        # final_rowscales
        

        # custom_bwd_inputs
        

        # custom_fwd_inputs score_mod
        g_softmax_bias: T.Buffer([1], accum_dtype), 


        dQ: T.Buffer(shape, accum_dtype), # type: ignore
        dK: T.Buffer(shape, dtype), # type: ignore
        dV: T.Buffer(shape, dtype), # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=thread_num) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            # should not store K to local if dim is large
            K_local = T.alloc_fragment([block_M, dim], dtype)
            # H100 wgmma
            # K_local_T = T.alloc_fragment([block_M, dim], dtype)
            V_local = T.alloc_fragment([block_M, dimv], dtype)
            V_shared = T.alloc_shared([block_M, dimv], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            # qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            # dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)

            # final_rowscales_declare
            

            # custom_bwd_declare
            

            # score_mod_declare
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT_0_0 = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_0_0_0 = T.alloc_fragment([block_M, block_N], accum_dtype)
            softmax_bias = T.alloc_fragment([1], accum_dtype)


            do = T.alloc_shared([block_N, dimv], dtype)
            dv = T.alloc_fragment([block_M, dimv], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            T.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                    # K_shared: tl.layout.make_swizzled_layout(K_shared),
                }
            )
            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
            T.copy(K_shared, K_local)
            # T.copy(K_shared, K_local_T)
            # T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_local)
            T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
            # custom_fwd_inputs_load_prolog
            softmax_bias[0] = g_softmax_bias[0]

            T.clear(dv)
            T.clear(dk)

            # TODO: is causal
            loop_st = T.floordiv(by * block_M, block_N) if is_casual else 0
            loop_ed = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                # T.gemm(K_local, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                if is_casual:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            by * block_M + i <= k * block_N + j, qkT[i, j], -T.infinity(qkT.dtype)
                        )
                # score_mod
                score_mod(qkT, qkT_0_0_0, softmax_bias) # qkT,

                # final_rowscales_load
                

                # online_func_fwd
                
                
                # TODO: is causal
                # if is_casual:
                #     for i, j in T.Parallel(block_M, block_N):
                #         qkT_0_0_0[i, j] = T.if_then_else(
                #             by * block_M + i <= k * block_N + j, qkT_0_0_0[i, j], 0
                #         )
                #     for i, j in T.Parallel(block_M, block_N):
                #         qkT[i, j] = T.if_then_else(
                #             by * block_M + i <= k * block_N + j, qkT[i, j], -1
                #         )
                
                T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                T.copy(qkT_0_0_0, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                # custom_bwd_inputs_load
                

                T.clear(dsT)
                # T.gemm(V_local, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # custom_bwd
                
                
                # score_mod_backward
                score_mod_backward(dsT, dsT_0_0, qkT, qkT_0_0_0, softmax_bias) #  qkT, 
                  
                                
                T.copy(dsT, dsT_cast)
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                # T.gemm(dsT_shared, K_local_T, dq, transpose_A=True)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    if k * block_N + i < seq_len:
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
            T.copy(dv, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
            T.copy(dk, dK[bz, by * block_M : (by + 1) * block_M, bx, :])

    return flash_bwd              
        

def flashattn_bwd_postprocess(batch, heads, seq_len, dim, dimv):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
        dQ: T.Buffer(shape, accum_dtype), # type: ignore
        dQ_out: T.Buffer(shape, dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return flash_bwd_post


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, *custom_fwd_inputs):
        BATCH, N_CTX, H, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        causal = True
        block_M = 128# 256# 128
        block_N = 128# 64# 128 # if D_HEAD <= 128 else 64
        stages = 2# 3
        thread_num = 256
        output_idx_list = [4]
        mod = tl.cached(kernel, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, block_M, block_N, stages, thread_num)
        if len(output_idx_list) == 1:
            o = mod(q, k, v, *custom_fwd_inputs)
            final_scale = []
        else:
            o, *final_scale = mod(q, k, v, *custom_fwd_inputs)
        ctx.save_for_backward(q, k, v, o, *custom_fwd_inputs, *final_scale)
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, *tmp = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD = q.shape
        D_HEAD_V = v.shape[-1]
        custom_fwd_inputs = tmp[:-0]
        final_rowscales = tmp[-0:]
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        block_M = 64
        block_N = 128 # 64 if D_HEAD <= 64 else 32
        mod_prep = tl.cached(flashattn_bwd_preprocess, [2], BATCH, H, N_CTX, D_HEAD, D_HEAD_V)
        mod_post = tl.cached(flashattn_bwd_postprocess, [1], BATCH, H, N_CTX, D_HEAD, D_HEAD_V)
        if False:
            delta = mod_prep(o, do)
        # TODO: causal
        is_casual = True
        output_idx_list = [5, 6, 7]
        mod = tl.cached(
            flashattn_bwd, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEAD_V, is_casual, block_M, block_N
        )
        if False:
            dq, dk, dv = mod(q, k, v, do, *tmp, delta)
        else:
            dq, dk, dv = mod(q, k, v, do, *tmp)
        dq = mod_post(dq)
        none_list = [None] * len(tmp)
        return dq, dk, dv, *none_list

attention = _attention.apply

CONFIG_FILE = "sigmoid.json"
TUNE_SPACE = {
    "block_M" : [64,128,192,256],
    "block_N" : [32,64,128,192,256],
    "stages" : [1,2,3],
    "num_threads" : [128,256],
}
def eval():
    BHSD = list(
        itertools.product(
            (1,64),
            (24,32),
            (2048,4096,16384, 32768),
            (64,128)
        )
    )
    # BHSD = [(64,24,16384,128)]
    print(BHSD)
    from benchmark.bench_utils import do_bench_sigmoidattn
    from autotuner.sigmoid_tunner import SigmoidTunner
    for B,H,S,D in BHSD:
        # do_bench_sigmoidattn(attention, B, H, S, D, D)

        st = SigmoidTunner(D,D,**TUNE_SPACE)
        configs = st.generate_config()
        st.tune(kernel, B,H,S,D,D,configs, CONFIG_FILE)


if __name__ == "__main__":
    D = 128
    B, H ,S, D = 1,24,32768,D
    from benchmark.bench_utils import do_bench_sigmoidattn
    do_bench_sigmoidattn(attention, B, H, S, D, D)

    # tune fwd: 
    from autotuner.sigmoid_tunner import SigmoidTunner
    st = SigmoidTunner(D,D,**TUNE_SPACE)
    configs = st.generate_config()
    st.tune(kernel, B,H,S,D,D,configs, CONFIG_FILE)

    # eval()

