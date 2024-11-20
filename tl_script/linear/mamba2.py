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

class SSD(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, decay, *custom_fwd_inputs):
        BATCH, H, N_CTX, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        BT = 64
        BK = 64
        BV = 64

        # decay_mod here

        decay_cumsum = chunk_local_cumsum_scalar(
            decay, BT
        )
        chunk_fwd_h_mod = tl.cached(chunk_fwd_h, [3,], BATCH, H, N_CTX, D_HEAD, D_HEADV, BT, BK, BV)
        output_idx_list = [5,]
        chunk_fwd_o_mod = tl.cached(chunk_o, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, BT, BK, BV)

        h = chunk_fwd_h_mod(k, v, decay_cumsum)
        o = chunk_fwd_o_mod(h.bfloat16(), q, k, v, decay_cumsum,*custom_fwd_inputs)

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

if __name__ == "__main__":
    torch.manual_seed(0)
    # torch.cuda.set_device(1)
    B, H, D, DV = 16, 8, 128, 128
    TLen = 16384 # 512
    BT= 64
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    X_mamba = 0.1 * torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    dt_mamba = 0.1*torch.randn(B, TLen, H, dtype=accum_dtype, device=device)
    A_mamba =  0.1*torch.rand(H, dtype=dtype, device=device)
    B_mamba =  0.1*torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    C_mamba = 0.1*torch.randn(B, TLen, H, D, dtype=dtype, device=device)
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

