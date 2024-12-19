import torch
from tvm import tl
import tvm.tl.language as T

import triton.language as triton_lang
import triton

import itertools

# ---------------------TL_KERNEL_CUMSUM
@triton.jit
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    T: triton_lang.constexpr,
    BT: triton_lang.constexpr,
):
    i_t, i_bh = triton_lang.program_id(0), triton_lang.program_id(1)
    p_s = triton_lang.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_o = triton_lang.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    # [BT, BS]
    b_s = triton_lang.load(p_s, boundary_check=(0,)).to(triton_lang.float32)
    b_o = triton_lang.cumsum(b_s, axis=0)
    triton_lang.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))

def chunk_local_cumsum_scalar(g, BT):
    B, H, T = g.shape
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        g_org, g,
        T=T, BT=BT
    )
    return g


# --------------- TL_KERNEL_H
def chunk_fwd_h(
        batch, headq, headk, head, seqlen, dim, dimv,
        BT, BK, BV, num_stages, num_threads
):
    # BT = 64
    # BK = 64
    # BV = 64
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = num_stages
    LOG2E = 1.44269504

    assert(head % headk == 0)
    head_headk_ratio = head // headk

    @T.prim_func
    def main(
        k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
        v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        h: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
    ):
        with T.Kernel(NK, NV, batch * head, threads=num_threads) as (bx, by, bz):

            b_h = T.alloc_fragment((BK, BV), accum_dtype)
            # b_h_cast = T.alloc_fragment((BK, BV), dtype)
            b_h_shared = T.alloc_shared((BK, BV), dtype)
            b_k = T.alloc_fragment((BT, BK), dtype)
            b_k_shared = T.alloc_shared((BT, BK), dtype)
            b_kt = T.alloc_fragment((BK, BT), dtype)
            b_v_shared = T.alloc_shared((BT, BV), dtype)
            b_v = T.alloc_fragment((BT, BV), dtype)
            b_g = T.alloc_fragment((BT), accum_dtype)
            b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            b_glast = T.alloc_fragment((1), accum_dtype)

            bhead = bz % head
            bb = bz // head

            bheadk = bhead // head_headk_ratio
            
            T.annotate_layout({
                # b_v_shared: tl.layout.make_swizzled_layout(b_v_shared),
                b_k_shared: tl.layout.make_swizzled_layout(b_k_shared),
                b_h_shared: tl.layout.make_swizzled_layout(b_h_shared),
            }
            )
            T.clear(b_h)

            for i_t in T.Pipelined(NT, num_stages=num_stages):
                # T.copy(b_h, b_h_shared)
                # T.copy(h[bb,bhead,(i_t*dim+bx*BK):(i_t*dim+(bx+1)*BK), by*BV:(by+1)*BV], b_h)
                # T.copy(k[bb,bhead,i_t*BT:(i_t+1)*BT,bx*BK:(bx+1)*BK], b_k)
                T.copy(k[bb,bheadk,i_t*BT:(i_t+1)*BT,bx*BK:(bx+1)*BK], b_k_shared)
                T.copy(v[bb,bhead,i_t*BT:(i_t+1)*BT,by*BV:(by+1)*BV], b_v_shared)
                T.copy(g[bb,bhead,i_t*BT:(i_t+1)*BT], b_g_shared)
                # T.copy(v[bb,bhead,i_t*BT:(i_t+1)*BT,by*BV:(by+1)*BV], b_v)
                # T.copy(b_v, b_v_shared)
                # T.copy(g[bb,bhead,i_t*BT:(i_t+1)*BT], b_g)
                # T.copy(g[bb,bhead,(i_t+1)*BT-1:(i_t+1)*BT], b_glast)
                b_glast[0] = g[bb,bhead,(i_t+1)*BT-1]

                T.copy(b_h, b_h_shared) # implicit cast
                T.copy(b_h_shared, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])
                # T.copy(b_h, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])

                # scalar_decay
                # for i0 in T.Parallel(BT):
                #     b_g[i0] = T.exp(b_g[i0])

                for i0, i1 in T.Parallel(BK, BV):
                    b_h[i0, i1] = b_h[i0, i1] * T.exp2(b_glast[0] * LOG2E)
                
                # for i0,i1 in T.Parallel(BT, BV):
                #     b_v_shared[i0,i1] *= T.exp2((b_glast[0] - b_g[i0]) * 1.44269504)

                T.copy(b_k_shared, b_k)
                T.copy(b_g_shared, b_g)
                for i0, i1 in T.Parallel(BK, BT):
                    b_kt[i0, i1] = b_k[i1, i0]*T.exp2((b_glast[0] - b_g[i1]) * LOG2E)

                T.gemm(b_kt, b_v_shared, b_h, transpose_B=False)
                # T.copy(b_h, b_h_shared)
                
                # TODO
                # T.copy(b_h, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])
    
    return main


# --------------- TL_KERNEL_O 
def chunk_o(
        batch, headq,headk, head, seqlen, dim, dimv,
        BT, BK, BV, num_stages, num_threads
):
    # BT = 64
    # BK = 64
    # BV = 64
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = num_stages
    LOG2E = 1.44269504

    scale = 1.0

    assert(head % headk == 0)
    head_headk_ratio = head // headk
    assert(head % headq == 0)
    head_headq_ratio = head // headq

    @T.prim_func
    def main(
        h: T.Buffer((batch,head,NT*dim,dimv), dtype), # type: ignore
        q: T.Buffer((batch,headq,seqlen,dim), dtype), # type: ignore
        k: T.Buffer((batch,headk,seqlen,dim), dtype), # type: ignore
        v: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
        g: T.Buffer((batch,head,seqlen), accum_dtype), # type: ignore
        o: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
        # custom fwd inputs
    ):
        with T.Kernel(NV, NT, batch * head, threads=num_threads) as (bx, by, bz):
            bo = T.alloc_fragment((BT, BV), dtype=accum_dtype)
            bo_shared = T.alloc_shared((BT, BV), dtype=dtype)
            bs = T.alloc_fragment((BT, BT), dtype=accum_dtype)
            bs_cast = T.alloc_fragment((BT, BT), dtype=dtype)
            bq = T.alloc_fragment((BT, BK), dtype=dtype)
            bq_shared = T.alloc_shared((BT,BK), dtype=dtype)
            bk_shared = T.alloc_shared((BT,BK), dtype=dtype)
            bv_shared = T.alloc_shared((BT,BV), dtype=dtype)
            b_state_shared = T.alloc_shared((BK, BV), dtype=dtype)
            bg_shared = T.alloc_shared((BT,), dtype=accum_dtype, scope = "shared")
            bg = T.alloc_fragment((BT,), dtype=accum_dtype)
            bg1 = T.alloc_fragment((BT,), dtype=accum_dtype)

            # custom fwd inputs init

            bb = bz // head
            bh = bz % head

            bhk = bh // head_headk_ratio
            bhq = bh // head_headq_ratio

            T.annotate_layout({
                bq_shared: tl.layout.make_swizzled_layout(bq_shared),
                bo_shared: tl.layout.make_swizzled_layout(bo_shared),
            })
            T.clear(bo)
            T.clear(bs)
            for ik in T.Pipelined(NK, num_stages=num_stages):
                # pipeline here
                T.copy(q[bb, bhq, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bq_shared)
                # T.copy(q[bb, bh, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bq)
                T.copy(k[bb, bhk, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bk_shared)

                T.copy(h[bb, bh, by*dim+ik*BK:by*dim+(ik+1)*BK, bx*BV:(bx+1)*BV], b_state_shared)
                
                T.copy(bq_shared, bq)
                # q_mod here (fused)
                {{q_mod_expr | indent(16)}}

                T.gemm(bq, bk_shared, bs, transpose_B=True)
                T.gemm(bq, b_state_shared, bo, transpose_B=False)
            
            # T.copy(g[bb, bh, by*BT:(by+1)*BT], bg)
            # T.copy(g[bb, bh, by*BT:(by+1)*BT], bg1)
            T.copy(g[bb, bh, by*BT:(by+1)*BT], bg_shared)
            T.copy(bg_shared, bg)
            T.copy(bg_shared, bg1)
            for i0,i1 in T.Parallel(BT,BV):
                bo[i0,i1] *= T.exp2(bg[i0] * LOG2E)
            
            for i0,i1 in T.Parallel(BT,BT):
                bs[i0,i1] *= T.exp2((bg[i0]-bg1[i1]) * LOG2E)
            # fix nan bug
            for i0,i1 in T.Parallel(BT,BT):
                bs[i0,i1] = T.if_then_else(
                    i0 >= i1, bs[i0,i1], 0.0
                )
            # tl bug here: 因为每个线程间的bg会有相同的
            # T.copy(bg,bg1)
            # for i0,i1 in T.Parallel(BT,BT):
            #     bs[i0,i1] *= T.exp2((bg[i0]-bg1[i1]) * LOG2E)
            T.copy(v[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV], bv_shared)
            T.copy(bs, bs_cast)
            T.gemm(bs_cast, bv_shared, bo)
            # T.copy(bo, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV]) # slow for stride between thread
            T.copy(bo, bo_shared) # implicit type convert
            T.copy(bo_shared, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV])
    
    return main

# --------------- TL_INTERFACE
class LinearAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, decay, {{custom_inputs_list}} *custom_fwd_inputs):
        BATCH, HQ, N_CTX, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        HK = k.shape[1]
        H = v.shape[1]

        # autotuner here
        # BT = 64
        # BK_h = 64
        # BV_h = 64
        # num_stages_h = 2
        # num_threads_h = 128
        # BK_o = 64
        # BV_o = 64
        # num_stages_o = 2
        # num_threads_o = 128
        BT = {{BT}}
        BK_h = {{BK_h}}
        BV_h = {{BV_h}}
        num_stages_h = {{num_stages_h}}
        num_threads_h = {{num_threads_h}}
        BK_o = {{BK_o}}
        BV_o = {{BV_o}}
        num_stages_o = {{num_stages_o}}
        num_threads_o = {{num_threads_o}}

        # decay_mod here
        {{decay_mod_expr | indent(8)}}

        # k_mod here
        {{k_mod_expr | indent(8)}}

        # v_mod here
        {{v_mod_expr | indent(8)}}

        decay_cumsum = chunk_local_cumsum_scalar(
            decay, BT
        )
        chunk_fwd_h_mod = tl.cached(chunk_fwd_h, [3,], BATCH, HQ,HK, H, N_CTX, D_HEAD, D_HEADV, BT, BK_h, BV_h, num_stages_h, num_threads_h)
        output_idx_list = [5,]
        chunk_fwd_o_mod = tl.cached(chunk_o, output_idx_list, BATCH, HQ,HK, H, N_CTX, D_HEAD, D_HEADV, BT, BK_o, BV_o, num_stages_o, num_threads_o)

        h = chunk_fwd_h_mod(k, v, decay_cumsum)
        o = chunk_fwd_o_mod(h, q, k, v, decay_cumsum,*custom_fwd_inputs)

        ctx.save_for_backward(q, k, v, decay_cumsum, {{custom_inputs_list}} *custom_fwd_inputs)
        ctx.BT = BT
        return o

    @staticmethod
    def backward(ctx, do):
        BT = ctx.BT
        q, k, v, decay_cumsum, {{custom_inputs_list}} *custom_fwd_inputs = ctx.saved_tensors
        # h = chunk_fwd_h_mod(k, v, decay_cumsum)
        pass

linear_attention = LinearAttention.apply

# --------------- TL_TUNNER
# AUTOtune
from autotuner.arch.H100 import H100

def generate_config(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT,device=H100()):
    # BTs = [32,64,128,192,256]
    BK_hs = [32,64,128,192,256]
    BV_hs = [32,64,128,192,256]
    num_stages_hs = [1,2,3,4]
    num_threads_hs = [128,256]
    BK_os = [32,64,128,256]
    BV_os = [32,64,128,256]
    num_stages_os = [1,2,3,4]
    num_threads_os = [128,256]
    
    # H100
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    # input shape
    # BTs = [bt for bt in BTs if N_CTX % bt == 0]
    BK_hs = [bk for bk in BK_hs if D_HEAD % bk == 0]
    BV_hs = [bv for bv in BV_hs if D_HEADV % bv == 0]
    BK_os = [bk for bk in BK_os if D_HEAD % bk == 0]
    BV_os = [bv for bv in BV_os if D_HEADV % bv == 0]
       
    config_h = []
    for BK_h, BV_h, num_stages_h, num_threads_h in itertools.product(BK_hs, BV_hs, num_stages_hs, num_threads_hs):
        sharedmem_chunk_h = 2 * (
            BK_h * BV_h + (BT * BK_h + BT * BV_h + BT ) 
            * num_stages_h)
        reg_chunk_h = 4 * (
            BK_h * BV_h
        )
        conditions_h = [
            num_stages_h > N_CTX // BT,
            BK_h % (MMA_ATOM_M) != 0,
            sharedmem_chunk_h > smem_cap,
            reg_chunk_h > reg_cap and reg_chunk_h > reg_cap_per_thread * num_threads_h,
        ]
        if any(conditions_h):
            continue
        config_h.append( {
        # "BT": BT,
        "BK_h": BK_h,
        "BV_h": BV_h,
        "num_stages_h": num_stages_h,
        "num_threads_h": num_threads_h,
        })
    
    config_o = []
    for BK_o, BV_o, num_stages_o, num_threads_o in itertools.product(BK_os, BV_os, num_stages_os, num_threads_os):
        sharedmem_chunk_o = 2 * (
            BT * BK_o * num_stages_o + BT * BK_o * num_stages_o + BK_o * BV_o* num_stages_o +
            BT * BV_o 
        )
        reg_chunk_o = 4 * (
            BT * BT + BT * BV_o
        )
        conditions_o = [
            num_stages_o > D_HEAD // BK_o,
            BT % (MMA_ATOM_M*(num_threads_o//MMA_ATOM_TRHEADS)) != 0,
            sharedmem_chunk_o > smem_cap,
            reg_chunk_o > reg_cap and reg_chunk_o > reg_cap_per_thread * num_threads_o,
        ]
        if any(conditions_o):
            continue
        config_o.append( {
        # "BT": BT,
        "BK_o": BK_o,
        "BV_o": BV_o,
        "num_stages_o": num_stages_o,
        "num_threads_o": num_threads_o,
        })
        
    return config_h, config_o

from autotuner.attnfwd_tunner_engine2 import tl_tune

def autotune(B, HQ, HK, H, Tlen, D, DV, file_path="mamba2",device=H100()):
            
    BTs = [32,64,128,192,256]
 
    best_config = {}
    best_latency = 1e6
    for BT in BTs:
        config_h,config_o = generate_config(B, HQ, HK, H, Tlen, D, DV, BT, device)
        if len(config_h) == 0 or len(config_o) == 0:
            continue
        problem_keys = {
            "B": B, "HQ": HQ, "HK": HK, "H": H, "Tlen": Tlen, "D": D, "DV": DV, "BT": BT
        }
        best_config_h, best_latency_h = tl_tune(chunk_fwd_h, problem_keys, config_h, [3,], file_path=f"{file_path}_h.json")
        best_config_o, best_latency_o  = tl_tune(chunk_o, problem_keys, config_o, [5,], file_path=f"{file_path}_o.json")
        if best_latency_h + best_latency_o < best_latency:
            best_latency = best_latency_h + best_latency_o
            best_config = {
                "BT": BT,
                "BK_h": best_config_h["BK_h"],
                "BV_h": best_config_h["BV_h"],
                "num_stages_h": best_config_h["num_stages_h"],
                "num_threads_h": best_config_h["num_threads_h"],
                "BK_o": best_config_o["BK_o"],
                "BV_o": best_config_o["BV_o"],
                "num_stages_o": best_config_o["num_stages_o"],
                "num_threads_o": best_config_o["num_threads_o"],
            }
    
    return best_config, best_latency
   
