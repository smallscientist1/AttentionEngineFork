from examples.mha import causal_softmax_attention
from examples.mha_v2 import causal_softmax_attention as causal_softmax_attention_v2
from attention_engine.benchmark.bench_utils import do_bench

import torch
import torch.nn.functional as F

def bench_fig11():
    
    Batches = [1,]
    seqlens = [2048, 4096, 8192]
    
    # (a) Softmax Attention (DeepSeek-V2-Lite)
    deepseek_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("causal_softmax_attn", b, 16, s, s, 192, 128)
        deepseek_data.append((f"BS{b}\nS{s}", result_dict))
    # TODO: decode
    # for b, s in [(B, S) for B in Batches for S in seqlens]:
    #     result_dict = bench_attention("causal_softmax_attn", b, 16, 1, s, 192, 128)
    #     deepseek_data.append((f"BS{b}S1\nKV{s}", result_dict))
    dump_bench_result("deepseek", deepseek_data)
        
    # (b) Softmax Attention (LLAMA-3.1-8B)
    llama_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("causal_softmax_attn", b, 32, s, s, 128, 128)
        llama_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("llama", llama_data)
        
    # (c) ReLU Attention (ViT-s/16-style)
    vit_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("relu_attn", b, 6, s, s, 64, 64)
        vit_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("vit", vit_data)
        
    # (d) Softmax Attention (Diff-Transformer-3B)
    dit_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("causal_softmax_attn", b, 12, s, s, 128, 256)
        dit_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("dit", dit_data)
        
    # (e) Retention Parallel (RetNet-6.7B)
    retnet_data = []
    for b, s in [(B, S) for B in Batches for S in [2048, 4096]]:
        result_dict = bench_attention("retention_parallel", b, 32, s, s, 256, 512)
        retnet_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("retnet", retnet_data)
    
    # (f) Sigmoid Attention (LLAMA-3.1-8B)
    sigmoid_attn_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("sigmoid_attn", b, 32, s, s, 128, 128)
        sigmoid_attn_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("sigmoid_attn", sigmoid_attn_data)
    
    # (h) Gated Retention (RFA-Big)
    rfa_data = []
    for b, s in [(B, S) for B in [64,] for S in [1024, 2048, 4096]]:
        result_dict = bench_attention("gated_retention", b, 16, s, s, 64, 64)
        rfa_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("rfa", rfa_data)
    
    # (g) Mamba2 SSM (Mamba2-2.7B)
    mamba2_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("mamba2_ssm", b, 1, s, s, 128, 64, head_v=80)
        mamba2_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("mamba2", mamba2_data)
    
    # (i) Gated Retention (YOCO-13B)
    yoco_data = []
    for b, s in [(B, S) for B in [8,] for S in [1024, 2048, 4096]]:
        result_dict = bench_attention("gated_retention", b, 40, s, s, 256, 256)
        yoco_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("yoco", yoco_data)
        
        
    # (j) RetNet Recurrent (RetNet-6.7B)
    retnet_recur_data = []
    for b, s in [(B, S) for B in Batches for S in [2048, 4096]]:
        result_dict = bench_attention("retention_recurrent", b, 32, s, s, 256, 512)
        retnet_recur_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("retnet_recur", retnet_recur_data)

    # (k) DeepSeek MLA
    mla_data = []
    for b, s in [(B, S) for B in [1,] for S in seqlens]:
        result_dict = bench_attention("mla_attn", b, 128, 1, s, 576, 512, head_k=1, head_v=1)
        mla_data.append((f"BS{b}S1\nKV{s}", result_dict))
    dump_bench_result("mla", mla_data)
    

    # (l) Sparse Group Query Attention
    sparse_gqa_data = []
    for b, s in [(B, S) for B in [1,] for S in seqlens]:
        result_dict = bench_attention("sparse_gqa", b, 32, 1, s, 128, 128, head_k=8, head_v=8)
        sparse_gqa_data.append((f"BS{b}S1\nKV{s}", result_dict))
    dump_bench_result("sparse_gqa", sparse_gqa_data)
    
    
def dump_bench_result(name:str, data):
    print(name, data)
    pass


def bench_attention(attn_type:str, Batch:int, head:int, seqlen_q:int, seqlen_kv:int, dim_qk:int, dim_v:int, head_k: int=None, head_v: int=None):
    if head_k is None:
        head_k = head
    if head_v is None:
        head_v = head
        
    if attn_type == "causal_softmax_attn":
        result_dict = bench_softmaxattention(Batch, head, seqlen_q, seqlen_kv, dim_qk, dim_v)
    # elif attn_type == "sigmoid_attn":
    #     result_dict = bench_sigmoidattention(Batch, head, seqlen_q, dim_qk, dim_v)
    else:
        # raise ValueError(f"Undefined attention type: {attn_type}")
        print("Warning: Undefined attention type, skipping benchmark.")
        result_dict = {}

    return result_dict

    
def bench_softmaxattention(B, H, Sq, S, D, DV, device='cuda', dtype=torch.float16, require_grad=True):
    
    # init input
    query = torch.randn(B, Sq, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, Sq, H, DV, device=device, dtype=dtype, requires_grad=False)

    result_dict = {}
    
    # ours
    attention_module = causal_softmax_attention(B, H, S, D, DV)
    def ours():
        o = attention_module(query, key, value)
        return o
    ours_fwd_lat = do_bench(ours)
    if require_grad:
        o = attention_module(query, key, value)
        ours_bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    result_dict["MetaAttention"] = (ours_fwd_lat, ours_bwd_lat)
    
    # FlashAttention-2
    fa2_lat = None
    try:
        from flash_attn import flash_attn_func
        def fa2(dim_padded):
            if D < dim_padded:
                query_padded = F.pad(query, (0, dim_padded - D), value=0.)
                key_padded = F.pad(key, (0, dim_padded - D), value=0.)
            else:
                query_padded = query
                key_padded = key
            if DV < dim_padded:
                value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
            else:
                value_padded = value
            o_ref = flash_attn_func(
                query_padded,
                key_padded,
                value_padded,
                softmax_scale=(
                    1 / D)**0.5,
                causal=True)
            if DV < dim_padded:
                o_ref = o_ref[:, :, :, :DV]
            return o_ref
        
        dim_padded_fa2 = max(D, DV)
        fa2_fwd_lat = do_bench(lambda: fa2(dim_padded_fa2))
        if require_grad:
            o_ref = fa2(dim_padded_fa2)
            fa2_bwd_lat = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        result_dict["FlashAttention-2"] = (fa2_fwd_lat, fa2_bwd_lat)
    except:
        print("Warning: FlashAttention-2 not available")
    
    # FlashAttention-3
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_func_hopper
        
        def fa3(dim_padded=0):
            if D < dim_padded:
                query_padded = F.pad(query, (0, dim_padded - D), value=0.)
                key_padded = F.pad(key, (0, dim_padded - D), value=0.)
            else:
                query_padded = query
                key_padded = key
            if DV < dim_padded:
                value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
            else:
                value_padded = value
            o_ref = flash_attn_func_hopper(
                query_padded, key_padded, value_padded, softmax_scale=(
                    1 / D)**0.5, causal=True)
            if DV < dim_padded:
                o_ref = o_ref[:, :, :, :DV]
            return o_ref
        
        dim_padded_fa3 = list(filter(lambda x: x >= max(D, DV), [64, 128, 192, 256]))
        # flash attention 3 specifically supported for D=192 and DV=128, so does not need padding for this case
        if D == 192 and DV == 128:
            dim_padded_fa3 = 0
        
        fa3_fwd_lat = do_bench(lambda: fa3(dim_padded_fa3))
        
        if require_grad:
            o_ref = fa3(dim_padded_fa3)
            fa3_bwd_lat = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        
        result_dict["FlashAttention-3"] = (fa3_fwd_lat, fa3_bwd_lat)
        
    except:
        print("Warning: FlashAttention-3 not available")
    
    return result_dict

def bench_sigmoidattention(B, H, S, D, DV):
    pass

def plot_fig():
    pass

if __name__ == "__main__":
    bench_fig11()
    plot_fig()
    

    
