import triton.language as triton_lang
import triton

# for ncu
import sys
import os
os.environ["PYTHONPATH"] = "/home/aiscuser/cfy/tvm/python:/home/aiscuser/cfy/flash-linear-attention"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/home/aiscuser/cfy/tvm/python")
sys.path.append("/home/aiscuser/cfy/flash-linear-attention")

import tvm.tl.language as T
from tvm import tl

from einops import rearrange, repeat

import torch

# g = 1: accuracy error! tl bug on A100: b_v_shared(cp.async,square), batch=2, NK=2, NV=2

# tl 0.9 ms 9.54 TFlops triton 0.73 ms 11.8 tflops

# tl 1.2 ms 28.7 tFLops block 64,64
# tl 1.08 ms 31.8 tFLops block 64,32
# stage 2: 1.12 ms 
# 0.93ms 37 TFlops tma_store
# 1.04ms ma pipeline
# 0.71ms 48 TFlops tma load k
# 0.64ms 54 TFLops swizzled layout
# 0.64ms h swizzle
# 0.47 ms 74 TFlops store h bf16

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


def chunk_fwd_h_ref(
        k, v, g
):
    # BK = 64
    BT = 64
    # BV = 64
    B, H, T, D = k.shape
    DV = v.shape[-1]
    NT = T // BT
    k = rearrange(k, "b h (t c) d -> b h t c d", c=BT)
    v = rearrange(v, "b h (t c) dv -> b h t c dv", c=BT)
    h = torch.zeros(B, H, NT, D, DV, dtype=torch.float32, device=k.device)
    h_tmp = torch.zeros(B, H, D, DV, dtype=torch.float32, device=k.device)
    for i in range(NT):

        h[:,:,i,:,:] = h_tmp

        g_last = g[:,:,(i+1)*BT-1].unsqueeze(-1).unsqueeze(-1)
        gg = g[:,:,i*BT:(i+1)*BT].unsqueeze(-1)
        gg = torch.exp(g_last - gg)
        h_tmp *= torch.exp(g_last)
        kg = k[:,:,i,:,:] * gg
        vg = v[:,:,i,:,:] * gg
        # h_tmp += torch.einsum("bhcd, bhcn -> bhdn", kg, v[:,:,i,:,:])
        h_tmp += torch.einsum("bhcd, bhcn -> bhdn", k[:,:,i,:,:].float(), vg).float()
        
    h = rearrange(h, "b h t d dv -> b h (t d) dv")
    return h.float()

def chunk_fwd_h_triton(
        k,v,g,
):
    BT=64
    from fla.ops.common.chunk_h import chunk_fwd_h_fn
    h,_ = chunk_fwd_h_fn(k, v, g, None, None, BT, None, output_final_state=False, states_in_fp32=False)
    return h

def chunk_fwd_h(
        batch, head, seqlen, dim, dimv,
        BT, BK, BV
):
    # BT = 64
    # BK = 64
    # BV = 64
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = 1

    @T.prim_func
    def main(
        k: T.Buffer((batch, head, seqlen, dim), dtype), # type: ignore
        v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        h: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
    ):
        with T.Kernel(NK, NV, batch * head, threads=128) as (bx, by, bz):

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

                # # v_mod
                # for i in T.Parallel():
                #     b_v_shared *= t_local[i]
                # 
                T.gemm(b_kt, b_v_shared, b_h, transpose_B=False)
                # T.copy(b_h, b_h_shared)
                
                # TODO
                # T.copy(b_h, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])
    
    return main

if __name__ == "__main__":
    torch.cuda.manual_seed(0)
    B, H, D, DV = 4, 16, 128, 128
    TLen = 16384 # 512
    BT, BK, BV = 64, 64, 32
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    final_state = False
    total_flops = 2*B * H * TLen * D * DV 

    q = torch.randn(B, H, TLen, D, dtype=dtype, device=device)
    k = torch.randn(B, H, TLen, D, dtype=dtype, device=device)
    v = torch.randn(B, H, TLen, DV, dtype=dtype, device=device)
    g = 0.1*torch.rand(B, H, TLen, dtype=accum_dtype, device=device)

    mod = tl.cached(chunk_fwd_h, [3,], B, H, TLen, D, DV, BT, BK, BV)
    # h = torch.zeros(B, H, (TLen//BT)*D, DV, dtype=torch.float32)
    h = mod(k,v,g).float() # mod(k, v, g)
    h_ref = chunk_fwd_h_ref(k, v, g)

    print(torch.allclose(h, h_ref, rtol=1e-3, atol=1e-3))
    print_debug(h, h_ref)
    
    # torch.testing.assert_close(h, h_ref, rtol=1e-3, atol=1e-3)
    # from tvm.tl.utils import do_bench
    # def run():
    #     h = mod(k,v,g) # chunk_fwd_h_triton(k,v,g) # mod(k, v, g)
    # def run_triton():
    #     h = chunk_fwd_h_triton(k, v, g)
    
    # latency = do_bench(run, warmup=500,rep=1000)
    # print("tl: {:.2f} ms".format(latency))
    # print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))

    # latency = do_bench(run_triton, warmup=500,rep=1000)
    # print("triton: {:.2f} ms".format(latency))
    # print("triton: {:.2f} TFlops".format(total_flops / latency * 1e-9))




        

