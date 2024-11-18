import torch
import triton
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.selective_state_update import selective_state_update

configs = [triton.testing.Benchmark(
            x_names=["batch", "seq_len", "nheads", "headdim", "ngroups", "dstate", "chunk_size"],
            x_vals=[
                (64, 4096, 64, 64, 8, 64, 256),
                ],
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=[""],
            line_names=[""],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="mamba2-performance-fp16",
            args={},
        )]

# @triton.testing.perf_report(configs)
# def benchmark(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
#     warmup = 25
#     rep = 100
#     x = torch.empty((batch, seq_len, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     dt = torch.empty((batch, nheads, seq_len // chunk_size, chunk_size), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     A = torch.empty((nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     B = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     C = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
   
#     quantiles = [0.5, 0.2, 0.8]
#     ms, min_ms, max_ms = triton.testing.do_bench(lambda: mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size), warmup=warmup, rep=rep, quantiles=quantiles)
#     flops_chunk_cumsum_fwd = 0
#     flops_chunk_state_fwd =  2.0 * batch * seq_len * nheads * headdim * dstate
#     flops_state_passing_fwd = 0
#     flops_bmm_chunk_fwd = 2.0 * batch * ngroups * dstate * seq_len * chunk_size
#     flops_chunk_scan_fwd = 2.0 * batch * seq_len * chunk_size * nheads * headdim + 2.0 * batch * seq_len * nheads * headdim * dstate
#     total_flops = flops_chunk_cumsum_fwd + flops_chunk_state_fwd + flops_state_passing_fwd + flops_bmm_chunk_fwd + flops_chunk_scan_fwd
#     perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=True, print_data=True)


# @triton.testing.perf_report(configs)
# def benchmark(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
#     warmup = 25
#     rep = 100
#     x = torch.empty((batch, seq_len, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     dt = torch.empty((batch, seq_len, nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     A = torch.empty((nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     B = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
#     C = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
   
#     quantiles = [0.5, 0.2, 0.8]
#     ms, min_ms, max_ms = triton.testing.do_bench(lambda: mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size), warmup=warmup, rep=rep, quantiles=quantiles)
#     print(ms)
#     flops_chunk_cumsum_fwd = 0
#     flops_chunk_state_fwd =  2.0 * batch * seq_len * nheads * headdim * dstate
#     flops_state_passing_fwd = 0
#     flops_bmm_chunk_fwd = 2.0 * batch * ngroups * dstate * seq_len * chunk_size
#     flops_chunk_scan_fwd = 2.0 * batch * seq_len * chunk_size * nheads * headdim + 2.0 * batch * seq_len * nheads * headdim * dstate
#     total_flops = flops_chunk_cumsum_fwd + flops_chunk_state_fwd + flops_state_passing_fwd + flops_bmm_chunk_fwd + flops_chunk_scan_fwd
#     perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=True, print_data=True)
# benchmark(64, 4096, 64, 64, 8, 64, 256, "")
# mean, max, min = benchmark(1, 1024, 64, 64, 1, 128, 256, "")
# print(mean, max, min)

# from transformers import AutoTokenizer, Mamba2Model
# import torch

# tokenizer = AutoTokenizer.from_pretrained("mistralai/mamba-codestral-7B-v0.1")
# model = Mamba2Model.from_pretrained("mistralai/mamba-codestral-7B-v0.1")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

def benchmark_chunk_cumsum(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd
    warmup = 25
    rep = 100
    dt = torch.empty((batch, seq_len, nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    A = torch.empty((nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
   
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: _chunk_cumsum_fwd(dt, A, chunk_size), warmup=warmup, rep=rep, quantiles=quantiles)
    print("chunk_cumsum:", ms)

def benchmark_chunk_state(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd
    warmup = 25
    rep = 100
    x = torch.empty((batch, seq_len, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    dt = torch.empty((batch, nheads, seq_len // chunk_size, chunk_size), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    B = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    dA_cumsum = torch.empty((batch, nheads, seq_len // chunk_size, chunk_size), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
   
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=True), warmup=warmup, rep=rep, quantiles=quantiles)
    flops_chunk_state_fwd =  2.0 * batch * seq_len * nheads * headdim * dstate
    perf = lambda ms: flops_chunk_state_fwd * 1e-12 / (ms * 1e-3)
    print("chunk_state:", ms, perf(ms))

def benchmark_state_passing(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
    warmup = 25
    rep = 100
    states = torch.empty((batch, seq_len // chunk_size, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    dA_chunk_cumsum = torch.empty((batch, nheads, seq_len // chunk_size), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    initial_states = torch.empty((batch, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
 
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: _state_passing_fwd(states, dA_chunk_cumsum, initial_states), warmup=warmup, rep=rep, quantiles=quantiles)
    print("state_passing:", ms)

def benchmark_bmm_chunk(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
    warmup = 25
    rep = 100
    A = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    B = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: _bmm_chunk_fwd(A, B, chunk_size), warmup=warmup, rep=rep, quantiles=quantiles)
    flops_bmm_chunk_fwd = 2.0 * batch * ngroups * dstate * seq_len * chunk_size
    perf = lambda ms: flops_bmm_chunk_fwd * 1e-12 / (ms * 1e-3)
    print("bmm_chunk:", ms, perf(ms))

def benchmark_chunk_scan(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
    warmup = 25
    rep = 100
    x = torch.empty((batch, seq_len, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    dt = torch.empty((batch, nheads, seq_len // chunk_size, chunk_size), device='cuda', dtype=torch.float32).normal_(-1.0, 1.0)
    C = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    dA_cumsum = torch.empty((batch, nheads, seq_len // chunk_size, chunk_size), device='cuda', dtype=torch.float32).normal_(-1.0, 1.0)
    states = torch.empty((batch, seq_len // chunk_size, nheads, headdim, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    cb = torch.empty((batch, seq_len // chunk_size, ngroups, chunk_size, chunk_size), device='cuda', dtype=torch.float32).normal_(-1.0, 1.0)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states), warmup=warmup, rep=rep, quantiles=quantiles)
    flops_chunk_scan_fwd = 2.0 * batch * seq_len * chunk_size * nheads * headdim + 2.0 * batch * seq_len * nheads * headdim * dstate
    perf = lambda ms: flops_chunk_scan_fwd * 1e-12 / (ms * 1e-3)
    print("chunk_scan:", ms, perf(ms))

def benchmark(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    warmup = 25
    rep = 100
    x = torch.empty((batch, seq_len, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    dt = torch.empty((batch, seq_len, nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    A = torch.empty((nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    B = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    C = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
   
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size), warmup=warmup, rep=rep, quantiles=quantiles)
    print("All:", ms)

def benchmark_mamba_chunk_scan_combined(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd
    warmup = 25
    rep = 100
    x = torch.empty((batch, seq_len, nheads, headdim), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    dt = torch.empty((batch, seq_len, nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    A = torch.empty((nheads), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    B = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
    C = torch.empty((batch, seq_len, ngroups, dstate), device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
   
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size), warmup=warmup, rep=rep, quantiles=quantiles)
    print("mamba_chunk_scan_combined:", ms)

def run(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider):
    # benchmark(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider)
    benchmark_mamba_chunk_scan_combined(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider)
    benchmark_chunk_cumsum(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider)
    benchmark_chunk_state(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider)
    benchmark_state_passing(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider)
    benchmark_bmm_chunk(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider)
    benchmark_chunk_scan(batch, seq_len, nheads, headdim, ngroups, dstate, chunk_size, provider)

run(64, 1024, 64, 64, 1, 128, 256, "")