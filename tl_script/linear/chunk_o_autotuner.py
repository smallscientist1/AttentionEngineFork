import triton.language as triton_lang
import triton

# for ncu
# import sys
# import os
# os.environ["PYTHONPATH"] = "/home/aiscuser/cfy/tvm/python:/home/aiscuser/cfy/flash-linear-attention"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# sys.path.append("/home/aiscuser/cfy/tvm/python")
# sys.path.append("/home/aiscuser/cfy/flash-linear-attention")

import tvm.tl.language as T
from tvm import tl
from tvm.tl.autotuner import *

from einops import rearrange, repeat
import itertools

import torch
from functools import partial
# import debugpy
# # debugpy.listen(5678)
# debugpy.connect(5678)

# accuracy totally correct
# torch.Size([2, 16, 16384, 64])
# tl: 0.20 ms
# tl: 105.48 TFlops
# triton: 0.19 ms
# triton: 113.22 TFlops
# torch.Size([2, 16, 512, 64])
# tl: 0.05 ms
# tl: 12.82 TFlops
# triton: 0.01 ms
# triton: 46.69 TFlops

# 4, 16, 16384, 128
# tl 0.68 ms  100.67 TFlops
# q_shared 0.65 ms 105.48 TFlops
# h_shared swizzle store 0.63 ms 109 TFlops
# stage2 0.58 ms 119.5 TFlops


LOG2E = 1.44269504

def print_debug(o, O_ref):
    close_mask = torch.isclose(o, O_ref, rtol=1e-3, atol=1e-3)
    total_elements = o.numel()
    num_not_close = (~close_mask).sum().item()
    percentage_not_close = (num_not_close / total_elements) * 100
    print(f"{percentage_not_close:.2f}% of the elements are not close.")
    print(f"Total elements: {total_elements}, Not close elements: {num_not_close}")
    # max diff and idx
    max_diff = (o - O_ref).abs().max().item()
    max_diff_idx = (o - O_ref).abs().argmax().item()
    max_diff_idx = torch.unravel_index(torch.tensor(max_diff_idx), o.shape)
    print(f"Max diff: {max_diff} at index {max_diff_idx}")
    print(f"Reference: {O_ref[max_diff_idx]}")
    print(f"Library: {o[max_diff_idx]}")
    print(torch.allclose(o, O_ref, rtol=1e-3, atol=1e-3))
    # max relative diff and idx
    max_rel_diff = ((o - O_ref).abs() / O_ref.abs()).max().item()
    max_rel_diff_idx = ((o - O_ref).abs() / O_ref.abs()).argmax().item()
    max_rel_diff_idx = torch.unravel_index(torch.tensor(max_rel_diff_idx), o.shape)
    print(f"Max rel diff: {max_rel_diff} at index {max_rel_diff_idx}")
    print(f"Reference: {O_ref[max_rel_diff_idx]}")
    print(f"Library: {o[max_rel_diff_idx]}")

    with open("o_ref.txt", "w") as f:
        O_ref_1 = O_ref.cpu()
        for idx, element in enumerate(O_ref_1):# .flatten()):
            f.write(f"{idx}: {element}\n")
    with open("o.txt", "w") as f:
        o_1 = o.cpu()
        for idx, element in enumerate(o_1):# .flatten()):
            f.write(f"{idx}: {element}\n")

def get_configs():
    # BT = [64]# [32, 64, 128]
    BK = [32, 64, 128]# [32, 64, 128]
    BV = [32, 64, 128]
    num_stages = [1, 2, 3]
    thread_num = [128, 256]
    _configs = list(itertools.product(BK, BV, num_stages, thread_num))

    configs = [
        {'BK': c[0], 'BV': c[1], 'num_stages': c[2], 'thread_num': c[3]}
        for c in _configs
    ]
    return configs

def chunk_o_triton(
        h,q,k,v,g,BT=64
):
    scale = 1.0
    from fla.ops.simple_gla.chunk import chunk_fwd_o_fn
    o = chunk_fwd_o_fn(
        h, q, k, v, g, BT, scale=scale
    )
    return o


def chunk_o(
        batch, head, seqlen, dim, dimv,
        BT, BK, BV,
):
    # BT = 64
    # BK = 64
    # BV = 64
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = 2

    scale = 1.0

    @T.prim_func
    def main(
        h: T.Buffer((batch,head,NT*dim,dimv), dtype), # type: ignore
        q: T.Buffer((batch,head,seqlen,dim), dtype), # type: ignore
        k: T.Buffer((batch,head,seqlen,dim), dtype), # type: ignore
        v: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
        g: T.Buffer((batch,head,seqlen), accum_dtype), # type: ignore
        o: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
    ):
        with T.Kernel(NV, NT, batch * head, threads=128) as (bx, by, bz):
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

def _chunk_o(
        batch, head, seqlen, dim, dimv,
        BT,
):
    @autotune(configs=get_configs(), keys=['BK', 'BV', 'num_stages', 'thread_num'], warmup=10, rep=20)
    @jit(out_idx=[5,], supply_type=tl.TensorSupplyType.Normal, ref_prog=partial(chunk_o_triton, BT=BT), skip_check=True)
    def kernel(
        BK=None, BV=None, num_stages=None, thread_num=None
    ):
        NT = seqlen // BT
        NK = dim // BK
        NV = dimv // BV
        dtype = "bfloat16"
        accum_dtype = "float"

        scale = 1.0

        @T.prim_func
        def main(
            h: T.Buffer((batch,head,NT*dim,dimv), dtype), # type: ignore
            q: T.Buffer((batch,head,seqlen,dim), dtype), # type: ignore
            k: T.Buffer((batch,head,seqlen,dim), dtype), # type: ignore
            v: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
            g: T.Buffer((batch,head,seqlen), accum_dtype), # type: ignore
            o: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
        ):
            with T.Kernel(NV, NT, batch * head, threads=thread_num) as (bx, by, bz):
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
                T.gemm(bs_cast, bv_shared, bo)
                T.copy(bs, bs_cast)
                # T.copy(bo, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV]) # slow for stride between thread
                T.copy(bo, bo_shared) # implicit type convert
                T.copy(bo_shared, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV])
    
        return main

    return kernel()


batches = [1]# , 16]
heads = [8, 64]
seqlens = [512,  16384]
BTs = [128]# [64,128] # 32 fail
DDVs = [(64,64),(128,128),(128,64),(256,256)]


if __name__ == "__main__":
    import csv
    torch.manual_seed(0)
    B, H, D, DV = 4, 16, 128, 128
    TLen = 16384
    BT, BK, BV = 64, 64, 64
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    f = open("chunk_fwd_o.csv", "w")
    writer = csv.writer(f)
    writer.writerow(["B", "H", "TLen", "D", "DV", "BT", "BK", "BV", "stage", "thread_num", "latency", "ref_latency", "TFlops"])
    f.flush()
    

    for B, H, TLen, BT, (D, DV) in itertools.product(batches, heads, seqlens, BTs, DDVs):
        print(f"B={B}, H={H}, TLen={TLen}, D={D}, DV={DV}, BT={BT}")
        total_flops = 2*B * H * TLen * BT * D + 2 * B * H * TLen * D * DV + 2 * B * H * TLen * BT * DV
        best_latency, best_config, ref_latency = _chunk_o(B, H, TLen, D, DV, BT)
        writer.writerow([B, H, TLen, D, DV, BT, best_config['BK'], best_config['BV'], best_config['num_stages'], best_config['thread_num'], best_latency, ref_latency, total_flops / best_latency * 1e-9])
        f.flush()
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
    
    # h = torch.randn(B, H, (TLen // BT) * D, DV, dtype=dtype, device=device)
    # q = torch.randn(B, H, TLen, D, dtype=dtype, device=device)
    # k = torch.randn(B, H, TLen, D, dtype=dtype, device=device)
    # v = torch.randn(B, H, TLen, DV, dtype=dtype, device=device)
    # g = torch.rand(B, H, TLen, dtype=accum_dtype, device=device)

    # mod = tl.cached(chunk_o, [5,], B, H, TLen, D, DV, BT, BK, BV)
    # o = mod(h, q, k, v, g)
    # o_ref = chunk_o_triton(h, q, k, v, g)
    # # print(o)
    # print_debug(o, o_ref)
    # print(o.shape)
    # # torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)

    # from tvm.tl.utils import do_bench
    # def run():
    #     o = mod(h, q, k, v, g) # chunk_fwd_h_triton(k,v,g) # mod(k, v, g)
    # def run_triton():
    #     o_ref = chunk_o_triton(h, q, k, v, g)
    
    # latency = do_bench(run, warmup=500,rep=1000)
    # print("tl: {:.2f} ms".format(latency))
    # print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))

    # latency = do_bench(run_triton, warmup=500,rep=1000)
    # print("triton: {:.2f} ms".format(latency))
    # print("triton: {:.2f} TFlops".format(total_flops / latency * 1e-9))
