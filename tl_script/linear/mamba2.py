import triton.language as triton_lang
import triton

import tvm.tl.language as T
from tvm import tl

from einops import rearrange, repeat

import torch
import torch.nn.functional as F

from chunk_h import chunk_fwd_h
from chunk_o import chunk_o, print_debug
from cumsum_kernel import chunk_local_cumsum_scalar

import math
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

# H100 4,16,128,128; 16384
# tl: 2.03 ms
# mamba2: 2.01ms

def check_close(o, O_ref, rtol=1e-3, atol=1e-3):
    absolute_error = torch.abs(o - O_ref)
    relative_error = absolute_error / (torch.abs(O_ref)+1e-6)

    # 检查是否满足任意一个条件
    tolerance_check = (absolute_error < atol) | (relative_error < rtol)

    # 打印不满足条件的百分比
    num_not_close = (~tolerance_check).sum().item()
    total_elements = o.numel()
    percentage_not_close = (num_not_close / total_elements) * 100
    print(f"{percentage_not_close:.2f}% of the elements are not close.")
    # 判断是否所有元素都满足任意一个条件
    result = torch.all(tolerance_check)
    if not result:
        # 打印不满足条件的百分比
        num_not_close = (~tolerance_check).sum().item()
        total_elements = o.numel()
        percentage_not_close = (num_not_close / total_elements) * 100
        print(f"{percentage_not_close:.2f}% of the elements are not close.")
        # 打印10个错误元素
        print("Error elements:")
        error_indices = torch.nonzero(~tolerance_check, as_tuple=False)
        for idx in error_indices[:10]:
            print(f"Index: {tuple(idx.cpu().numpy())}, Reference: {O_ref[tuple(idx.cpu().numpy())]}, Library: {o[tuple(idx.cpu().numpy())]}")

    return result

class SSD(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, decay, *custom_fwd_inputs):
        BATCH, H, N_CTX, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        BT = 64
        BK_h = 64
        BV_h = 64
        BK_o = 64
        BV_o = 64

        # decay_mod here

        # k_mod here

        # v_mod here
    #    h = exp(decay)*h+ k@v
    #    o = q@h

        decay_cumsum = chunk_local_cumsum_scalar(
            decay, BT
        )
        chunk_fwd_h_mod = tl.cached(chunk_fwd_h, [3,], BATCH, H, N_CTX, D_HEAD, D_HEADV, BT, BK_h, BV_h)
        output_idx_list = [5,]
        chunk_fwd_o_mod = tl.cached(chunk_o, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, BT, BK_o, BV_o)

        h = chunk_fwd_h_mod(k, v, decay_cumsum)
        o = chunk_fwd_o_mod(h, q, k, v, decay_cumsum,*custom_fwd_inputs)

        ctx.save_for_backward(q, k, v, decay_cumsum, *custom_fwd_inputs)
        ctx.BT = BT
        return o

    @staticmethod
    def backward(ctx, do):
        BT = ctx.BT
        q, k, v, decay_cumsum, *custom_fwd_inputs = ctx.saved_tensors
        # h = chunk_fwd_h_mod(k, v, decay_cumsum)
        pass

ssd = SSD.apply

batches = [1]# , 16]
heads = [8, 128]
seqlens = [512,  16384]
BTs = [64, 128]
DDVs = [(64,64),(128,128)]

if __name__ == "__main__":
    torch.cuda.manual_seed(0)
    # torch.cuda.set_device(1)
    B, H, D, DV = 16, 8, 128, 128
    TLen = 16384 # 512
    BT= 64
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    X_mamba = 0.4 * torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    dt_mamba = 0.7*torch.randn(B, TLen, H, dtype=accum_dtype, device=device)
    A_mamba =  1.5*torch.randn(H, dtype=dtype, device=device) - 4
    B_mamba =  0.8 * torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    C_mamba = torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    # q = torch.randn(B, H, TLen, D, dtype=dtype, device=device)
    # k = torch.randn(B, H, TLen, D, dtype=dtype, device=device)
    # v = torch.randn(B, H, TLen, DV, dtype=dtype, device=device)
    # g = 0.1*torch.rand(B, H, TLen, dtype=accum_dtype, device=device)

        # initialize dt
    factory_kwargs = {"device": device, "dtype": dtype}
    dt_min=0.001
    dt_max=0.1
    dt = torch.exp(
            torch.rand(H, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)
    print(dt_bias_mamba.max().item(), dt_bias_mamba.min().item())
    print(dt_mamba.max().item(), dt_mamba.min().item())
    print(A_mamba.max().item(), A_mamba.min().item())

    out_ref = mamba_chunk_scan_combined(
        X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
        chunk_size=BT, D=None, return_final_states=False,
        dt_bias=None# dt_bias_mamba
    )
    # print(out_ref)

    q = C_mamba.transpose(1, 2).contiguous()
    k = (B_mamba * dt_mamba[...,None]).transpose(1, 2).bfloat16().contiguous()
    v = X_mamba.transpose(1, 2).contiguous()
    # g = (A_mamba * dt_mamba).transpose(1, 2).contiguous()
    A_mamba1 = A_mamba[:, None].contiguous()
    dt_bias_mamba1 = dt_bias_mamba[:, None].contiguous()
    dt_mamba1 = dt_mamba.transpose(1, 2).contiguous()
    out = ssd(
        q, k, v, dt_mamba1*A_mamba1 # F.softplus(dt_mamba + dt_bias_mamba) * A_mamba
    )
    out = out.transpose(1, 2).contiguous()

    print_debug(out, out_ref)
    # torch.testing.assert_close(out, out_ref, rtol=1e-1, atol=1e-1)
    assert check_close(out, out_ref, rtol=5e-2, atol=5e-2)

    from tvm.tl.utils import do_bench
    def run():
        out = ssd(
            q, k, v, dt_mamba1*A_mamba1
        )
    def run_ref():
        out_ref = mamba_chunk_scan_combined(
            X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
            chunk_size=BT, D=None, return_final_states=False,
            dt_bias=None# dt_bias_mamba
        )
    
    do_bench(run)
    do_bench(run_ref)

    latency = do_bench(run, warmup=500,rep=1000)
    print("tl: {:.2f} ms".format(latency))

    latency = do_bench(run_ref, warmup=500,rep=1000)
    print("MAMBA2: {:.2f} ms".format(latency))

