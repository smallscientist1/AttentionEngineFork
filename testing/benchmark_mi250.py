from examples.mha import causal_softmax_attention
from examples.reluattn import relu_attention
from examples.retnet_recurrent import retnet_recurrent
from examples.mamba2 import mamba2
from examples.mla_decode import mla_decode


# from attention_engine.benchmark.bench_utils import do_bench
from tilelang.profiler import do_bench

import torch
import torch.nn.functional as F
import math
import triton
import pandas as pd
import os

RESULT_DIR = "./results_mi250"
os.makedirs(RESULT_DIR, exist_ok=True)

def bench_fig12():
    
    Batches = [1,8]
    seqlens = [2048, 4096, 8192]
    
    # (a) Softmax Attention (DeepSeek-V2-Lite)
    deepseek_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("causal_softmax_attn", b, 16, s, s, 192, 128)
        deepseek_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("deepseek", deepseek_data)
    
    # (c) ReLU Attention (ViT-s/16-style)
    vit_data = []
    for b, s in [(B, S) for B in [32,64] for S in [512, 1024, 2048]]:
        result_dict = bench_attention("relu_attn", b, 6, s, s, 64, 64)
        vit_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("vit", vit_data)
    
    # (g) Mamba2 SSM (Mamba2-2.7B)
    mamba2_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("mamba2_ssm", b, 1, s, s, 128, 64, head_v=80)
        mamba2_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("mamba2", mamba2_data)
        
    # (j) RetNet Recurrent (RetNet-6.7B)
    retnet_recur_data = []
    for b, s in [(B, S) for B in Batches for S in [2048, 4096]]:
        result_dict = bench_attention("retention_recurrent", b, 32, s, s, 256, 512)
        retnet_recur_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("retnet_recur", retnet_recur_data)

    # (k) DeepSeek MLA
    mla_data = []
    for b, s in [(B, S) for B in [8,] for S in seqlens]:
        result_dict = bench_attention("mla_attn", b, 128, 1, s, 576, 512, head_k=1, head_v=1)
        mla_data.append((f"BS{b}S1\nKV{s}", result_dict))
    dump_bench_result("mla", mla_data)
    
    
def dump_bench_result(name: str, data: list):
    """
    将 benchmark 数据导出为 fwd 和 bwd 两个 CSV 文件。
    行(Index)为方法名 (如 MetaAttention)，列(Columns)为配置 (如 BS1\\nS2048)。
    """
    print(name, data)
    if not data:
        return
    
    # 1. 准备容器
    # 我们使用字典来构建数据，结构为: { method_name: { config_name: value } }
    fwd_data_dict = {}
    bwd_data_dict = {}
    
    # 用于保持列的顺序（即 data 中配置出现的顺序）
    config_order = []

    # 2. 解析数据
    for config_name, metrics in data:
        if config_name not in config_order:
            config_order.append(config_name)
            
        # metrics 可能是 None，或者是个空字典
        if not metrics:
            continue

        for method_name, values in metrics.items():
            # 初始化该方法的字典（如果尚未存在）
            if method_name not in fwd_data_dict:
                fwd_data_dict[method_name] = {}
            if method_name not in bwd_data_dict:
                bwd_data_dict[method_name] = {}
            
            # 处理数值
            # values 可能是 None，或者是一个元组 (fwd_val, bwd_val)
            if values is None:
                fwd_val, bwd_val = None, None
            else:
                # 安全解包，防止 tuple 长度不对
                fwd_val = values[0] if len(values) > 0 else None
                bwd_val = values[1] if len(values) > 1 else None
            
            fwd_data_dict[method_name][config_name] = fwd_val
            bwd_data_dict[method_name][config_name] = bwd_val

    # 3. 转换为 DataFrame
    # orient='index' 表示字典的键（Method Name）作为行索引
    df_fwd = pd.DataFrame.from_dict(fwd_data_dict, orient='index')
    df_bwd = pd.DataFrame.from_dict(bwd_data_dict, orient='index')

    # 4. 重新排列列顺序
    # 确保 CSV 的列顺序与 input data list 中的顺序一致
    # 可能会有某些配置在 data 中存在但在某些 method 中缺失，pandas 会自动填 NaN
    # 这里取交集以防万一 data 中有重复 key 或者 DataFrame 没生成的列
    valid_cols = [c for c in config_order if c in df_fwd.columns]
    df_fwd = df_fwd[valid_cols]
    
    valid_cols_bwd = [c for c in config_order if c in df_bwd.columns]
    df_bwd = df_bwd[valid_cols_bwd]

    # 5. 导出 CSV
    # 替换换行符，防止 CSV 格式在某些编辑器中显示混乱（可选，这里保留原样，CSV标准支持换行）
    # 如果你想把表头的换行去掉，可以取消下面注释:
    df_fwd.columns = df_fwd.columns.astype(str).str.replace('\n', ' ')
    df_bwd.columns = df_bwd.columns.astype(str).str.replace('\n', ' ')

    fwd_filename = os.path.join(RESULT_DIR, f"{name}_fwd.csv")
    bwd_filename = os.path.join(RESULT_DIR, f"{name}_bwd.csv")

    df_fwd.to_csv(fwd_filename, index_label="Method")
    df_bwd.to_csv(bwd_filename, index_label="Method")

    print(f"已保存: {fwd_filename}")
    print(f"已保存: {bwd_filename}")


def bench_attention(attn_type:str, Batch:int, head:int, seqlen_q:int, seqlen_kv:int, dim_qk:int, dim_v:int, head_k: int=None, head_v: int=None, require_grad: bool=True):
    if head_k is None:
        head_k = head
    if head_v is None:
        head_v = head
        
    result_dict = {}
    if attn_type == "causal_softmax_attn":
        result_dict = bench_softmaxattention(Batch, head, seqlen_q, seqlen_kv, dim_qk, dim_v, require_grad=require_grad)
    elif attn_type == "relu_attn":
        result_dict = bench_reluattention(Batch, head, seqlen_q, dim_qk, dim_v)
    elif attn_type == "retention_recurrent":
        result_dict = bench_retnet_recurrent(Batch, head, seqlen_q, dim_qk, dim_v)
    elif attn_type == "mamba2_ssm":
        result_dict = bench_mamba2_ssm(Batch, head, seqlen_q, dim_qk, dim_v, HK=head_k, HV=head_v)
    elif attn_type == "mla_attn":
        result_dict = bench_mla_decode(Batch, head, seqlen_kv, dim_qk, dim_v, HKV=head_k)
    else:
        # raise ValueError(f"Undefined attention type: {attn_type}")
        print(f"Warning: Undefined attention type {attn_type}, skipping benchmark.")
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
    if Sq < S:
        # decode
        assert require_grad == False
        attention_module = softmax_attention_decode(B, H, Sq, S, D, DV)
    else:
        attention_module = causal_softmax_attention(B, H, S, D, DV, tune=True)
    def ours():
        o = attention_module(query, key, value)
        return o
    ours_fwd_lat = do_bench(ours)
    if require_grad:
        o = attention_module(query, key, value)
        ours_bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    else:
        ours_bwd_lat = None
    
        
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
        else:
            fa2_bwd_lat = None
        result_dict["FlashAttention-2"] = (fa2_fwd_lat, fa2_bwd_lat)
    except Exception as e:
        print(f"Warning: FlashAttention-2 not available: {e}")

    return result_dict

def bench_reluattention(B, H, S, D, DV, device='cuda', dtype=torch.float16, require_grad=True):
    
    result_dict = {}
    query = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=False)
    
    # ours
    attention_module = relu_attention(B, H, S, D, DV, dtype=dtype, tune=True)
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

def bench_retnet_recurrent(B, H, S, D, DV, device="cuda", dtype=torch.bfloat16, require_grad=True):
    
    result_dict = {}
    # prepare input
    accum_dtype = torch.float32
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    g = torch.tensor(range(0, H), dtype=accum_dtype)
    g = 1 - torch.exp2(-5 - g)
    g = g[None, :, None].expand(B, H, S).cuda().detach().contiguous()
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

def bench_mamba2_ssm(B, HQ, S, D, DV, HK=None, HV=None, dtype=torch.bfloat16, require_grad=True):
    
    result_dict = {}
    
    # init input
    query = torch.randn(B, S, HQ, D, device="cuda", dtype=dtype)
    key = torch.randn(B, S, HK, D, device="cuda", dtype=dtype)
    value = torch.randn(B, S, HV, DV, device="cuda", dtype=dtype)
    if require_grad:
        do = 0.1 * torch.randn(B, S, HV,
                                     DV, dtype=dtype, device="cuda")
    A_mamba = 1.5 * torch.randn(HV, dtype=dtype, device="cuda") - 4.0
    # initialize dt
    accum_dtype = torch.float32
    dt_mamba = 0.7 * torch.randn(B, S, HV, dtype=accum_dtype, device="cuda")
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    dt_min = 0.001
    dt_max = 0.1
    dt = torch.exp(
        torch.rand(HV, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)
    
    # ours
    q_ours = query.clone().transpose(1, 2).contiguous()
    k_ours = key.clone().transpose(1, 2).contiguous()
    v_ours = value.clone().transpose(1, 2).contiguous()
    A_ours = A_mamba[None, :].clone().contiguous()
    dt_ours = dt_mamba.clone().transpose(1, 2).contiguous()
    if require_grad:
        do_ours = do.clone().transpose(1, 2).contiguous()
    
    q_ours = q_ours.detach().requires_grad_(require_grad)
    k_ours = k_ours.detach().requires_grad_(require_grad)
    v_ours = v_ours.detach().requires_grad_(require_grad)
    A_ours = A_ours.detach().requires_grad_(require_grad)
    dt_ours = dt_ours.detach().requires_grad_(require_grad)
    
    attention_module = mamba2(B, HQ, S, D, DV, HK, HV, dtype=dtype, tune=True)
    fwd_lat = do_bench(
        lambda: attention_module(q_ours, k_ours, v_ours, dt_ours, A_ours, dt_ours.to(dtype))
    )
    if require_grad:
        o = attention_module(q_ours, k_ours, v_ours, dt_ours, A_ours, dt_ours.to(dtype))
        bwd_lat = do_bench(lambda: o.backward(do_ours, retain_graph=True))
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
        
    # mamba2 ssm
    value = value.detach_().requires_grad_(require_grad)
    A_mamba.detach_().requires_grad_(require_grad)
    dt_mamba.detach_().requires_grad_(require_grad)
    key.detach_().requires_grad_(require_grad)
    query.detach_().requires_grad_(require_grad)

    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        fwd_lat_ref = do_bench(
            lambda: mamba_chunk_scan_combined(
                value,
                dt_mamba,
                A_mamba,
                key,
                query,
                chunk_size=64,
            )
        )
        if require_grad:
            out_ref = mamba_chunk_scan_combined(
                value,
                dt_mamba,
                A_mamba,
                key,
                query,
                chunk_size=64,
            )
            bwd_lat_ref = do_bench(lambda: out_ref.backward(do, retain_graph=True))
        result_dict["Mamba2SSM"] = (fwd_lat_ref, bwd_lat_ref)
    except Exception as e:
        print(f"Warning: mamba2 ssm not available: {e}")

    return result_dict

def bench_mla_decode(B, HQ, SKV, D, DV, HKV=1, dtype=torch.float16):
    result_dict = {}

    q = torch.randn(B, 1, HQ, DV, dtype=dtype, device="cuda")
    q_pe = torch.randn(B, 1, HQ, D-DV, dtype=dtype, device="cuda")
    KV = torch.randn(B, SKV, HKV, DV, dtype=dtype, device="cuda")
    k_pe = torch.randn(B, SKV, HKV, D-DV, dtype=dtype, device="cuda")
    
    # ours
    attention_module = mla_decode(B, HQ, SKV, D, DV, HK=HKV, HV=HKV, dtype=dtype)
    fwd_lat = do_bench(lambda: attention_module(q, q_pe, KV, k_pe))
    result_dict["MetaAttention"] = (fwd_lat, None)
    
    # flashMLA
    try:
        from ref.flash_mla_decode_triton import flash_mla_triton
        
        cache_seqlens = torch.full((B,), SKV, dtype=torch.int32, device="cuda")
        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
        
        block_size = 64
        block_table = torch.arange(
            B * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda"
        ).view(B, max_seqlen_pad // block_size)
        
        # [B, S, H, D] -> [B*S//64, 64, H, D]
        KV = KV.view(B * SKV // block_size, block_size, HKV, DV)
        k_pe = k_pe.view(B * SKV // block_size, block_size, HKV, D - DV)
        fwd_lat_ref = do_bench(lambda: flash_mla_triton(
            q, q_pe,
            block_table,
            KV, k_pe,
            max_seqlen_pad,
            block_size,
            B, 1, cache_seqlens, HQ, HKV, D, DV, True, dtype)
        )

        result_dict["FlashMLA"] = (fwd_lat_ref, None)

    except Exception as e:
        print(f"Warning: flashMLA not available: {e}")
        
    return result_dict


def plot_fig():
    pass

if __name__ == "__main__":
    import time
    start_time = time.time()
    bench_fig12()
    print(f"Benchmarking completed in {time.time() - start_time:.2f} seconds")
    plot_fig()
    

    
