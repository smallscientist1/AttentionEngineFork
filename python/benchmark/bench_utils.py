import torch
import math
import torch.nn.functional as F

import torch
import matplotlib.pyplot as plt

def analysis_tensor_data(a: torch.Tensor, plot: bool = False, figure_name: str = 'tensor_distribution.png'):
    print(f"Data type: {a.dtype}")
    print(f"Device: {a.device}")
    print(f"Shape: {a.shape}")
    
    max_value = a.max()
    min_value = a.min()
    mean_value = a.mean()
    std_value = a.std()
    median_value = a.median()
    # quantiles = torch.quantile(a.float(), torch.tensor([0.25, 0.5, 0.75], device=a.device))
    histogram = torch.histc(a.float(), bins=50, min=0, max=1)

    if plot:
        plt.hist(a.float().cpu().detach().numpy().flatten(), bins=50)
        plt.title('Tensor Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig(figure_name)
    
    print(f"Max: {max_value}")
    print(f"Min: {min_value}")
    print(f"Mean: {mean_value}")
    print(f"Std: {std_value}")
    print(f"Median: {median_value}")
    # print(f"Quantiles 4: {quantiles}")
    print(f"Histogram: {histogram}")

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

def print_debug(o, O_ref, rtol=1e-3, atol=1e-3):
    close_mask = torch.isclose(o, O_ref, rtol=rtol, atol=atol)
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
    print(torch.allclose(o, O_ref, rtol=rtol, atol=atol))
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

def do_bench_mamba(linear_attention, B, H, TLen, D, DV, BT):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    torch.cuda.manual_seed(0)
    # torch.cuda.set_device(1)
    # B, H, D, DV = 16, 8, 128, 128
    # TLen = 16384 # 512
    # BT= 64
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

    # initialize dt_bias
    factory_kwargs = {"device": device, "dtype": dtype}
    dt_min=0.001
    dt_max=0.1
    dt = torch.exp(
            torch.rand(H, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))

    # compute dt_mamba(not fused !!)
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)

    # print(dt_bias_mamba.max().item(), dt_bias_mamba.min().item())
    # print(dt_mamba.max().item(), dt_mamba.min().item())
    # print(A_mamba.max().item(), A_mamba.min().item())

    out_ref = mamba_chunk_scan_combined(
        X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
        chunk_size=BT, D=None, return_final_states=False,
        dt_bias=None# dt_bias_mamba
    )
    # print(out_ref)

    q = C_mamba.transpose(1, 2).contiguous()
    k = B_mamba.transpose(1, 2).contiguous() # (B_mamba * dt_mamba[...,None]).transpose(1, 2).bfloat16().contiguous()
    v = X_mamba.transpose(1, 2).contiguous()
    # g = (A_mamba * dt_mamba).transpose(1, 2).contiguous()
    A_mamba1 = A_mamba[None,:].clone().contiguous()
    dt_bias_mamba1 = dt_bias_mamba[:, None].contiguous()
    dt_mamba1 = dt_mamba.clone().transpose(1, 2).contiguous()
    dt_mamba1_k = dt_mamba1.clone().bfloat16()# .contiguous()
    print(dt_mamba1.shape, A_mamba1.shape)
    out = linear_attention(
        q, k, v, dt_mamba1, A_mamba1 , dt_mamba1_k
    )
    out = out.transpose(1, 2).contiguous()

    print_debug(out, out_ref)
    # torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=4e-2)
    assert check_close(out, out_ref, rtol=5e-2, atol=4e-2)

    from tvm.tl.utils import do_bench
    def run():
        out = linear_attention(
            q, k, v, dt_mamba1, A_mamba1 , dt_mamba1_k
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

def test_mamba_simple_gla():
    B, H, TLen, D, DV, BT = 16, 8, 512, 128, 128, 64
    from fla.ops.simple_gla import chunk_simple_gla
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    
    torch.cuda.manual_seed(0)

    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"

    X_mamba = 0.4 * torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    X_mamba = X_mamba.detach().requires_grad_()
    dt_mamba = 0.7*torch.randn(B, TLen, H, dtype=accum_dtype, device=device)
    dt_mamba = dt_mamba.detach().requires_grad_()    
    A_mamba =  1.5*torch.randn(H, dtype=dtype, device=device) - 4
    A_mamba = A_mamba.detach().requires_grad_()
    B_mamba =  0.8 * torch.randn(B, TLen, H, D, dtype=dtype, device=device)
    B_mamba = B_mamba.detach().requires_grad_()
    C_mamba = torch.randn(B, TLen, H, D, dtype=dtype, device=device, requires_grad=True)
    do_mamba = 0.1*torch.randn(B, TLen, H, DV, dtype=dtype, device=device)

    factory_kwargs = {"device": device, "dtype": dtype}
    dt_min=0.001
    dt_max=0.1
    dt = torch.exp(
            torch.rand(H, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))

    # compute dt_mamba(not fused !!)
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)
    # tmp: test!!! nan
    # dt_mamba = 0.7*torch.randn(B, TLen, H, dtype=accum_dtype, device=device)
    dt_mamba = dt_mamba.detach().requires_grad_()

    out_ref = mamba_chunk_scan_combined(
        X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
        chunk_size=BT, D=None, return_final_states=False,
        dt_bias=None# dt_bias_mamba
    )
    out_ref.backward(do_mamba)

    # print(out_ref)


    q = C_mamba.clone().detach().transpose(1, 2).contiguous().requires_grad_()
    k = B_mamba.clone().detach().transpose(1, 2).contiguous().requires_grad_()
    v = X_mamba.clone().detach().transpose(1, 2).contiguous().requires_grad_()
    # g = (A_mamba * dt_mamba).transpose(1, 2).contiguous()
    A_mamba1 = A_mamba[None,:].clone().detach().contiguous().requires_grad_()
    dt_bias_mamba1 = dt_bias_mamba[:, None].contiguous()
    dt_mamba1 = dt_mamba.clone().detach().transpose(1, 2).contiguous().requires_grad_()
    dt_mamba1_k = dt_mamba1.clone().detach().bfloat16().requires_grad_()
    
    do_mamba1 = do_mamba.transpose(1, 2).contiguous()
    out,_ = chunk_simple_gla(
        # gla dt_mamba1 is bf16
        q, k*dt_mamba1_k[...,None], v, dt_mamba1_k*A_mamba1[...,None],
        scale=1.0, output_final_state=False
    )
    out.backward(do_mamba1)
    out = out.transpose(1, 2).contiguous()

    print_debug(out, out_ref)
    # print("out_ref", out_ref)
    # print("out", out)
    # analysis_tensor_data(out_ref, plot=True, figure_name='tensor_distribution_out_ref.png')
    # analysis_tensor_data(out, plot=True, figure_name='tensor_distribution_out.png')

    # torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=4e-2)
    assert check_close(out, out_ref, rtol=5e-2, atol=4e-2)

    # print("q.grad", q.grad)
    # print("C_mamba.grad", C_mamba.grad)
    q_grad = q.grad.transpose(1, 2)
    k_grad = k.grad.transpose(1, 2)
    v_grad = v.grad.transpose(1, 2)
    dt_mamba1_grad = dt_mamba1_k.grad.transpose(1, 2)
    A_mamba1_grad = A_mamba1.grad.squeeze(0)

    assert check_close(q_grad, C_mamba.grad, rtol=1e-2, atol=4e-2)
    assert check_close(k_grad, B_mamba.grad, rtol=5e-2, atol=4e-2)
    assert check_close(v_grad, X_mamba.grad, rtol=5e-2, atol=4e-2)
    # here ???
    assert check_close(dt_mamba1_grad, dt_mamba.grad, rtol=5e-2, atol=5e-1)
    # print("dt_mamba1_grad", dt_mamba1_grad)
    # print("dt_mamba.grad", dt_mamba.grad)
    # analysis_tensor_data(dt_mamba1_grad, plot=True, figure_name='tensor_distribution_dt_mamba1_grad01.png')
    # analysis_tensor_data(dt_mamba.grad, plot=True, figure_name='tensor_distribution_dt_mamba_grad01.png')
    assert check_close(A_mamba1_grad, A_mamba.grad, rtol=9e-2, atol=4e-2)

    from tvm.tl.utils import do_bench
    def run():
        out,_ = chunk_simple_gla(
            q, k*dt_mamba1_k[...,None], v, dt_mamba1_k*A_mamba1[...,None],
            scale=1.0, output_final_state=False
        )
        out.backward(do_mamba1)
    def run_ref():
        out_ref = mamba_chunk_scan_combined(
            X_mamba, dt_mamba, A_mamba, B_mamba, C_mamba,
            chunk_size=BT, D=None, return_final_states=False,
            dt_bias=None# dt_bias_mamba
        )
        out_ref.backward(do_mamba)

    do_bench(run)
    do_bench(run_ref)

    print("fwd+bwd: ")
    latency = do_bench(run, warmup=500,rep=1000)
    print("triton: {:.2f} ms".format(latency))

    latency = do_bench(run_ref, warmup=500,rep=1000)
    print("MAMBA2: {:.2f} ms".format(latency))


def do_bench_simple_gla(linear_attention, B, H, TLen, D, DV, BT):
    torch.cuda.manual_seed(0)
    # torch.cuda.set_device(1)
    # B, H, D, DV = 16, 8, 128, 128
    # TLen = 16384 # 512
    # BT= 64
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"
    require_grad = True

    q = torch.randn(B, H, TLen, D, device=device, requires_grad=require_grad, dtype=dtype)
    k = torch.randn(B, H, TLen, D, device=device, requires_grad=require_grad, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, H, TLen, device=device, dtype=accum_dtype)).clamp_min(-5).requires_grad_(require_grad)
    v = torch.randn(B, H, TLen, DV, device=device, requires_grad=require_grad, dtype=dtype)

    do = torch.randn(B, H, TLen, DV, device=device, dtype=dtype)

    from tvm.tl.utils import do_bench
    def run():
        out = linear_attention(
            q, k, v, g
        )
    from fla.ops.simple_gla import chunk_simple_gla
    g1 = g.clone().bfloat16().requires_grad_(require_grad)
    def run_ref():
        out,_ = chunk_simple_gla(
            q, k, v, g1, scale=None, output_final_state=False
        )
    
    do_bench(run)
    do_bench(run_ref)

    latency = do_bench(run, warmup=500,rep=1000)
    print("tl: {:.2f} ms".format(latency))
    latency = do_bench(run_ref, warmup=500,rep=1000)
    print("simple gla: {:.2f} ms".format(latency))

def do_bench_retention_linear(linear_attention, B, H, TLen, D, DV):
    torch.cuda.manual_seed(0)
    # torch.cuda.set_device(1)
    # B, H, D, DV = 16, 8, 128, 128
    # TLen = 16384 # 512
    # BT= 64
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"
    require_grad = True

    q = torch.randn(B, H, TLen, D, device=device, requires_grad=require_grad, dtype=dtype)
    k = torch.randn(B, H, TLen, D, device=device, requires_grad=require_grad, dtype=dtype)
    g = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), H, dtype=accum_dtype))).detach()
    g = g[None, :, None].expand(B, H, TLen).cuda()
    v = torch.randn(B, H, TLen, DV, device=device, requires_grad=require_grad, dtype=dtype)

    do = torch.randn(B, H, TLen, DV, device=device, dtype=dtype)

    from tvm.tl.utils import do_bench
    def run():
        out = linear_attention(
            q, k, v, g
        )
    from fla.ops.retention import chunk_retention
    def run_ref():
        out,_ = chunk_retention(
            q, k, v, scale=None, output_final_state=False
        )
    
    do_bench(run)
    do_bench(run_ref)

    latency = do_bench(run, warmup=500,rep=1000)
    print("tl: {:.2f} ms".format(latency))
    latency = do_bench(run_ref, warmup=500,rep=1000)
    print("retention: {:.2f} ms".format(latency))


def do_bench_sigmoidattn(attn, B, H, S, D, DV):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = torch.float16
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    softmax_bias = torch.randn(1, device=device, dtype=torch.float)

    o = attn(query, key, value, softmax_bias)
    # print(o)
    o.backward(do, retain_graph=True)
    # print(query.grad)
    # print(key.grad)
    # print(value.grad)

    from tvm.tl.utils import do_bench
    def run():
        o = attn(query, key, value, softmax_bias)
    def run_bacward():
        o.backward(do, retain_graph=True)
    
    do_bench(run)
    do_bench(run_bacward)

    latency = do_bench(run, warmup=500,rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops/latency*1e-9))

    latency = do_bench(run_bacward, warmup=500,rep=1000)
    print("tl bwd: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(bwd_tflops/latency*1e-9))

def do_bench_retention(attn, B, H, S, D, DV):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = torch.float16
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    mask = torch.randn(
        1, H, S, S, device="cuda", dtype=torch.float16, requires_grad=False
    ).tril().contiguous()

    o = attn(query, key, value, mask)
    # print(o)
    # o.backward(do, retain_graph=True)
    # print(query.grad)
    # print(key.grad)
    # print(value.grad)

    from tvm.tl.utils import do_bench
    def run():
        o = attn(query, key, value, mask)
    # def run_bacward():
    #     o.backward(do, retain_graph=True)
    
    do_bench(run)
    # do_bench(run_bacward)

    latency = do_bench(run, warmup=500,rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops/latency*1e-9))

    # latency = do_bench(run_bacward, warmup=500,rep=1000)
    # print("tl bwd: {:.2f} ms".format(latency))
    # print("tflops: {:.2f}".format(bwd_tflops/latency*1e-9))

def do_bench_attention(attn, B, H, S, D, DV):
    tflops = 2 * B * H * S * S * D + 2 * B * H * S * S * DV
    tflops = tflops * 0.5
    bwd_tflops = 4 * B * H * S * S * DV + 6 * B * H * S * S * D
    bwd_tflops = bwd_tflops * 0.5
    torch.cuda.manual_seed(0)
    dtype = torch.float16
    device = "cuda"
    accum_dtype = torch.float32
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=True
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )
    do = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=True
    )

    o = attn(query, key, value)
    # print(o)
    o.backward(do, retain_graph=True)
    # print(query.grad)
    # print(key.grad)
    # print(value.grad)

    from tvm.tl.utils import do_bench
    def run():
        o = attn(query, key, value)
    def run_bacward():
        o.backward(do, retain_graph=True)
    
    do_bench(run)
    do_bench(run_bacward)

    latency = do_bench(run, warmup=500,rep=1000)
    print("tl: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(tflops/latency*1e-9))

    latency = do_bench(run_bacward, warmup=500,rep=1000)
    print("tl bwd: {:.2f} ms".format(latency))
    print("tflops: {:.2f}".format(bwd_tflops/latency*1e-9))


