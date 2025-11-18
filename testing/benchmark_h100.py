from examples.mha import causal_softmax_attention
from examples.mha_v2 import causal_softmax_attention as causal_softmax_attention_v2
from examples.gated_retention import gated_retention
from examples.sigmoid_attn import sigmoid_attention
from examples.reluattn_v2 import relu_attention
from examples.retnet_recurrent import retnet_recurrent
from examples.retention_parallel import retention_parallel

# from attention_engine.benchmark.bench_utils import do_bench
from tilelang.profiler import do_bench

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
        
    result_dict = {}
    if attn_type == "causal_softmax_attn":
        pass
        # result_dict = bench_softmaxattention(Batch, head, seqlen_q, seqlen_kv, dim_qk, dim_v)
    # elif attn_type == "sigmoid_attn":
    #     result_dict = bench_sigmoidattention(Batch, head, seqlen_q, dim_qk, dim_v)
    elif attn_type == "gated_retention":
        result_dict = bench_gated_retention(Batch, head, seqlen_q, dim_qk, dim_v)
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

def bench_sigmoidattention(B, H, S, D, DV, dtype=torch.float16, require_grad=True):
    
    result_dict = {}
    
    accum_dtype = torch.float32
    query = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device="cuda", dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, S, H, DV, device="cuda", dtype=dtype, requires_grad=False)
    
    softmax_bias = 0.1 * torch.randn(1, device="cuda", dtype=accum_dtype, requires_grad=False)
    
    softmax_bias_2 = softmax_bias.to("cpu")
    
    # ours
    attention_module = sigmoid_attention(B, H, S, D, DV)
    fwd_lat = do_bench(lambda: attention_module(query, key, value, softmax_bias))
    if require_grad:
        o = attention_module(query, key, value, softmax_bias)
        bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
    
    # flash-sigmoid
    try:
        from flash_sigmoid import flash_attn_func
        
        fwd_lat_ref = do_bench(lambda: flash_attn_func(
            query,
            key,
            value,
            softmax_scale=1.0,
            causal=True,
            sigmoid_bias=softmax_bias_2))
        if require_grad:
            out_ref = flash_attn_func(
                query,
                key,
                value,
                softmax_scale=1.0,
                causal=True,
                sigmoid_bias=softmax_bias_2)
            bwd_lat_ref = do_bench(lambda: out_ref.backward(do, retain_graph=True))
            
        result_dict["FlashSigmoid"] = (fwd_lat_ref, bwd_lat_ref)
    except Exception:
        print("Warning: flash-sigmoid not available")
    
    return result_dict
    
def bench_reluattention(B, H, S, D, DV, device='cuda', dtype=torch.float16, require_grad=True):
    
    result_dict = {}
    query = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=False)
    
    # ours
    attention_module = relu_attention(B, H, S, D, DV, dtype=dtype)
    fwd_lat = do_bench(lambda: attention_module(query, key, value))
    if require_grad:
        o = attention_module(query, key, value)
        bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
    
    # Pytorch ReLU Attention
    def ref_program(query, key, value):
        qk = torch.einsum('bqhd,bkhd->bhqk', query, key)
        qk = qk / (D ** 0.5)
        qk = F.relu(qk)
        o = torch.einsum('bhqk,bkhd->bqhd', qk, value)
        return o

    ref_program_fwd_lat = do_bench(lambda: ref_program(query, key, value))
    if require_grad:
        out_ref = ref_program(query, key, value)
        ref_program_bwd_lat = do_bench(lambda: out_ref.backward(do, retain_graph=True))
    result_dict["PytorchReLU"] = (ref_program_fwd_lat, ref_program_bwd_lat)
    
    return result_dict

def bench_gated_retention(B, H, S, D, DV, device='cuda', dtype=torch.bfloat16, require_grad=True):
    
    result_dict = {}
    # prepare input
    accum_dtype = torch.float32
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    g = F.logsigmoid(torch.randn(B, H, S, device="cuda", dtype=accum_dtype)).clamp_min(-5)
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    
    q.detach_().requires_grad_(require_grad)
    k.detach_().requires_grad_(require_grad)
    g.detach_().requires_grad_(require_grad)
    v.detach_().requires_grad_(require_grad)


    q1 = q.clone()
    k1 = k.clone()
    v1 = v.clone()
    g1 = g.clone().to(dtype)
    
    q1.detach_().requires_grad_(require_grad)
    k1.detach_().requires_grad_(require_grad)
    g1.detach_().requires_grad_(require_grad)
    v1.detach_().requires_grad_(require_grad)
    
    # ours
    attention_module = gated_retention(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = do_bench(lambda: attention_module(q, k, v, g))
    if require_grad:
        o = attention_module(q, k, v, g)
        bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
    
    # flash-linear-attention
    # try:
    from fla.ops.simple_gla import chunk_simple_gla
    fwd_lat_ref = do_bench(lambda: chunk_simple_gla(
        q1, k1, v1, g1, head_first=True
    )[0])
    
    if require_grad:
        out_ref,_ = chunk_simple_gla(
            q1, k1, v1, g1, head_first=True
        )
        bwd_lat_ref = do_bench(lambda: out_ref.backward(do, retain_graph=True))
    result_dict["FlashLinearAttention"] = (fwd_lat_ref, bwd_lat_ref)
    
    # except:
    #     print("Warning: fla.ops.simple_gla not available")
    #     return
    
    return result_dict
   
def bench_retnet_recurrent(B, H, S, D, DV, device="cuda", dtype=torch.bfloat16, require_grad=True):
    
    result_dict = {}
    # prepare input
    accum_dtype = torch.float32
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    g = torch.tensor(range(0, H), dtype=accum_dtype)
    g = 1 - torch.exp2(-5 - g)
    g = g[None, :, None].expand(B, H, TLen).cuda().detach().contiguous()
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    
    q.detach_().requires_grad_(require_grad)
    k.detach_().requires_grad_(require_grad)
    g.detach_().requires_grad_(False)
    v.detach_().requires_grad_(require_grad)

    # clone for reference
    q1 = q.clone()
    k1 = k.clone()
    v1 = v.clone()
    g1 = g.clone()

    q1.detach_().requires_grad_(require_grad)
    k1.detach_().requires_grad_(require_grad)
    g1.detach_().requires_grad_(False)
    v1.detach_().requires_grad_(require_grad)

    # ours
    attention_module = retnet_recurrent(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = do_bench(lambda: attention_module(q, k, v, g))
    if require_grad:
        o = attention_module(q, k, v, g)
        bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
    
    # flash-linear-attention
    try:
        from fla.ops.retention import chunk_retention
        fwd_lat_ref = do_bench(lambda: chunk_retention(
            q1, k1, v1, head_first=True
        )[0])
        if require_grad:
            o_ref, _ = chunk_retention(
                q1, k1, v1, head_first=True
            )
            bwd_lat_ref = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        result_dict["FlashLinearAttention"] = (fwd_lat_ref, bwd_lat_ref)
    except Exception:
        print("Warning: fla.ops.retention not available")
                
    
    return result_dict

def bench_retention_parallel(B, H, S, D, DV, device="cuda", dtype=torch.float16, require_grad=False):
    
    result_dict = {}
    # prepare input
    accum_dtype = torch.float32
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=require_grad)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=require_grad)
    v = torch.randn(B, S, H, DV, device="cuda", dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, S, H, DV, device="cuda", dtype=dtype, requires_grad=False)
    mask = torch.rand(
        1, H, S, S, device="cuda", dtype=dtype, requires_grad=False
    ).tril().contiguous()


    # ours
    attention_module = retention_parallel(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = do_bench(lambda: attention_module(q, k, v, mask))
    
    result_dict["MetaAttention"] = (fwd_lat, None)
    
    # pytorch 
    
    @torch.compile
    def ref_program(q, k, v, mask):
        qk = torch.einsum('bqhd,bkhd->bhqk', q, k)
        qkm = qk * mask
        r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        o = torch.einsum('bhqk,bkhd->bqhd', qkm / r, v)
        return o.to(dtype=dtype)

    ref_lat = do_bench(lambda: ref_program(q, k, v, mask))
    result_dict["PytorchRetention"] = (ref_lat, None)
    
    return result_dict


def plot_fig():
    pass

if __name__ == "__main__":
    bench_fig11()
    plot_fig()
    

    
