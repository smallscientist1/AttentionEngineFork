import jinja2

TL_IMPORT = """
import torch
from tvm import tl
import tvm.tl.language as T

import triton.language as triton_lang
import triton
"""

TL_KERNEL_CUMSUM = """
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
"""

TL_KERNEL_H = """
def chunk_fwd_h(
        batch, head, seqlen, dim, dimv,
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

    @T.prim_func
    def main(
        k: T.Buffer((batch, head, seqlen, dim), dtype), # type: ignore
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
            b_glast = T.alloc_fragment((1), accum_dtype)

            bhead = bz % head
            bb = bz // head
            
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
                T.copy(k[bb,bhead,i_t*BT:(i_t+1)*BT,bx*BK:(bx+1)*BK], b_k_shared)
                T.copy(v[bb,bhead,i_t*BT:(i_t+1)*BT,by*BV:(by+1)*BV], b_v_shared)
                # T.copy(v[bb,bhead,i_t*BT:(i_t+1)*BT,by*BV:(by+1)*BV], b_v)
                # T.copy(b_v, b_v_shared)
                T.copy(g[bb,bhead,i_t*BT:(i_t+1)*BT], b_g)
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
                for i0, i1 in T.Parallel(BK, BT):
                    b_kt[i0, i1] = b_k[i1, i0]*T.exp2((b_glast[0] - b_g[i1]) * LOG2E)

                T.gemm(b_kt, b_v_shared, b_h, transpose_B=False)
                # T.copy(b_h, b_h_shared)
                
                # TODO
                # T.copy(b_h, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])
    
    return main
"""

TL_KERNEL_O = """
def chunk_o(
        batch, head, seqlen, dim, dimv,
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

    @T.prim_func
    def main(
        h: T.Buffer((batch,head,NT*dim,dimv), dtype), # type: ignore
        q: T.Buffer((batch,head,seqlen,dim), dtype), # type: ignore
        k: T.Buffer((batch,head,seqlen,dim), dtype), # type: ignore
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
            # bg_shared = T.alloc_shared((BT,), dtype=accum_dtype)
            bg = T.alloc_fragment((BT,), dtype=accum_dtype)
            bg1 = T.alloc_fragment((BT,), dtype=accum_dtype)

            # custom fwd inputs init

            bb = bz // head
            bh = bz % head

            T.annotate_layout({
                bq_shared: tl.layout.make_swizzled_layout(bq_shared),
                bo_shared: tl.layout.make_swizzled_layout(bo_shared),
            })
            T.clear(bo)
            T.clear(bs)
            for ik in T.Pipelined(NK, num_stages=num_stages):
                # pipeline here
                T.copy(q[bb, bh, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bq_shared)
                # T.copy(q[bb, bh, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bq)
                T.copy(k[bb, bh, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bk_shared)

                T.copy(h[bb, bh, by*dim+ik*BK:by*dim+(ik+1)*BK, bx*BV:(bx+1)*BV], b_state_shared)
                
                T.copy(bq_shared, bq)
                # q_mod here (fused)
                {{q_mod_expr | indent(16)}}

                T.gemm(bq, bk_shared, bs, transpose_B=True)
                T.gemm(bq, b_state_shared, bo, transpose_B=False)
            
            T.copy(g[bb, bh, by*BT:(by+1)*BT], bg)
            T.copy(g[bb, bh, by*BT:(by+1)*BT], bg1)
            # TL BUG: deadlock here
            # T.copy(g[bb, bh, by*BT:(by+1)*BT], bg_shared)
            # T.copy(bg_shared, bg)
            # T.copy(bg_shared, bg1)
            for i0,i1 in T.Parallel(BT,BV):
                bo[i0,i1] *= T.exp2(bg[i0] * LOG2E)
            
            for i0,i1 in T.Parallel(BT,BT):
                bs[i0,i1] = T.if_then_else(
                    i0 >= i1, bs[i0,i1], 0.0
                )
            # tl bug here: 因为每个线程间的bg会有相同的
            # T.copy(bg,bg1)
            # for i0,i1 in T.Parallel(BT,BT):
            #     bs[i0,i1] *= T.exp2((bg[i0]-bg1[i1]) * LOG2E)
            for i0,i1 in T.Parallel(BT,BT):
                bs[i0,i1] *= T.exp2((bg[i0]-bg1[i1]) * LOG2E)
            T.copy(v[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV], bv_shared)
            T.copy(bs, bs_cast)
            T.gemm(bs_cast, bv_shared, bo)
            # T.copy(bo, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV]) # slow for stride between thread
            T.copy(bo, bo_shared) # implicit type convert
            T.copy(bo_shared, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV])
    
    return main
"""

TL_INTERFACE = """
class LinearAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, decay, {{custom_inputs_list}} *custom_fwd_inputs):
        BATCH, H, N_CTX, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        # autotuner here
        BT = 64
        BK_h = 64
        BV_h = 64
        num_stages_h = 2
        num_threads_h = 128
        BK_o = 64
        BV_o = 64
        num_stages_o = 1
        num_threads_o = 128

        # decay_mod here
        {{decay_mod_expr | indent(8)}}

        # k_mod here
        {{k_mod_expr | indent(8)}}

        # v_mod here

        decay_cumsum = chunk_local_cumsum_scalar(
            decay, BT
        )
        chunk_fwd_h_mod = tl.cached(chunk_fwd_h, [3,], BATCH, H, N_CTX, D_HEAD, D_HEADV, BT, BK_h, BV_h, num_stages_h, num_threads_h)
        output_idx_list = [5,]
        chunk_fwd_o_mod = tl.cached(chunk_o, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, BT, BK_o, BV_o, num_stages_o, num_threads_o)

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
"""

class TlLinearAttnTemplate:
    def __init__(self, **kargs):
        template = jinja2.Template(TL_IMPORT + \
                        TL_KERNEL_CUMSUM + \
                        TL_KERNEL_H + \
                        TL_KERNEL_O + \
                        TL_INTERFACE)

        kargs = {k:(v if v is not None else "") for k,v in kargs.items() }
        self.tlcode = template.render(**kargs)

    def __call__(self):
        return self.tlcode