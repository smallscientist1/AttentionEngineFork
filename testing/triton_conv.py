import torch

import triton
import triton.language as tl


def is_cuda():
    # return triton.runtime.driver.active.get_current_target().backend == "cuda"
    return True


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_autotune_config():
    return get_cuda_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'H', 'W', 'F', 'K', 'S', 'D', 'P'],
)
@triton.jit
def conv2d_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        N, C, H, W, F, K, S, D, P,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        # stride_an, stride_ac, stride_ah, stride_aw, #
        stride_am, stride_ak,
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    KH, KW = K, K
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N * OH * OW, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(F, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    aptr_base = a_ptr + (offs_am[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(KH * KW * C, BLOCK_SIZE_K)):
        m = offs_am
        p = m % (OH * OW) // OW
        q = m % OW
        r = (k * BLOCK_SIZE_K) // (KW * C)
        s = ((k * BLOCK_SIZE_K) // C) % KW
        h = p[:, None] * S + r[None, :] * D - P
        w = q[:, None] * S + s[None, :] * D - P
        mask_x = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        a_ptrs = aptr_base + ((k * BLOCK_SIZE_K) % C)
        a = tl.load(a_ptrs, mask=mask_x, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < KH * KW * C - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < N * OH * OW) & (offs_cn[None, :] < F)
    tl.store(c_ptrs, c, mask=c_mask)


def conv2d(a, b, S, D, P):
    N, H, W, C = a.shape
    KH, KW, C, F = b.shape
    assert KH == KW
    K = KH
    a = a.view(N * H * W, C)
    b = b.view(KH * KW * C, F)
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
    c = torch.empty((N * OH * OW, F), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(F, META['BLOCK_SIZE_N']) * triton.cdiv(N * OH * OW, META['BLOCK_SIZE_M']), )
    conv2d_kernel[grid](
        a, b, c,  #
        N, C, H, W, F, K, S, D, P,
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c.view(N, OH, OW, F)

def ref_program(A, B, stride, padding, dilation):
    A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
    B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
    C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
    C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
    return C

# torch.manual_seed(0)
# # N, C, H, W, F, K, S, D, P = 1, 256, 7, 7, 256, 1, 1, 1, 0
# N, C, H, W, F, K, S, D, P = 1, 1, 8, 8, 1, 3, 1, 1, 1
# # N, C, H, W, F, K, S, D, P = 128,256,56,56,64,1,1,1,0
# # N, C, H, W, F, K, S, D, P = 1, 16, 7, 7, 16, 1, 1, 1, 0
# # a = torch.randint(low=-2, high=3, size=(N, H, W, C), device='cuda', dtype=torch.float16)
# # b = torch.randint(low=-2, high=3, size=(K, K, C, F), device='cuda', dtype=torch.float16)
# a = torch.ones(N, H, W, C, device='cuda', dtype=torch.float16)
# b = torch.ones(K, K, C, F, device='cuda', dtype=torch.float16)
# triton_output = conv2d(a, b, S, D, P)
# torch_output = ref_program(a, b, stride=S, padding=P, dilation=D)
# rtol = 0
# if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
#     print("✅ Triton and Torch match")
# else:
#     # count how many percent not match within 1e-2
#     diff = torch.abs(triton_output - torch_output)
#     not_match = torch.count_nonzero(diff > 1e-2)
#     total = torch.numel(diff)
#     print(f"❌ Triton and Torch differ by {not_match}/{total} ({not_match/total*100:.2f}%)")
#     print("triton_output:", triton_output)
#     print("torch_output:", torch_output)

fp8_inputs = False
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N", "C", "H", "W", "F", "K", "S", "D", "P"],  # Argument names to use as an x-axis for the plot
        x_vals=[
            (1, 512, 7, 7, 2048, 1, 1, 1, 0),
            (1, 512, 14, 14, 512, 3, 2, 1, 1),
            (1, 1024, 14, 14, 512, 1, 1, 1, 0),
            (1, 256, 14, 14, 1024, 1, 1, 1, 0),
            (1, 256, 28, 28, 256, 3, 2, 1, 1),
            (1, 512, 28, 28, 256, 1, 1, 1, 0),
            (1, 128, 28, 28, 512, 1, 1, 1, 0),
            (1, 256, 56, 56, 128, 1, 1, 1, 0),
            (1, 64, 56, 56, 256, 1, 1, 1, 0),
            (1, 64, 56, 56, 64, 3, 1, 1, 1),
            (1, 64, 56, 56, 64, 1, 1, 1, 0),
            (1, 256, 56, 56, 64, 1, 1, 1, 0),
            (1, 256, 56, 56, 512, 1, 2, 1, 0),
            (1, 128, 28, 28, 128, 3, 1, 1, 1),
            (1, 512, 28, 28, 128, 1, 1, 1, 0),
            (1, 512, 28, 28, 1024, 1, 2, 1, 0),
            (1, 256, 14, 14, 256, 3, 1, 1, 1),
            (1, 1024, 14, 14, 256, 1, 1, 1, 0),
            (1, 1024, 14, 14, 2048, 1, 2, 1, 0),
            (1, 512, 7, 7, 512, 3, 1, 1, 1),
            (1, 2048, 7, 7, 512, 1, 1, 1, 0),
            (1, 128, 56, 56, 128, 3, 2, 1, 1),
            (1, 3, 224, 224, 64, 7, 2, 1, 3),
            (128, 512, 7, 7, 2048, 1, 1, 1, 0),
            (128, 512, 14, 14, 512, 3, 2, 1, 1),
            (128, 1024, 14, 14, 512, 1, 1, 1, 0),
            (128, 256, 14, 14, 1024, 1, 1, 1, 0),
            (128, 256, 28, 28, 256, 3, 2, 1, 1),
            (128, 512, 28, 28, 256, 1, 1, 1, 0),
            (128, 128, 28, 28, 512, 1, 1, 1, 0),
            (128, 256, 56, 56, 128, 1, 1, 1, 0),
            (128, 64, 56, 56, 256, 1, 1, 1, 0),
            (128, 64, 56, 56, 64, 3, 1, 1, 1),
            (128, 64, 56, 56, 64, 1, 1, 1, 0),
            (128, 256, 56, 56, 64, 1, 1, 1, 0),
            (128, 256, 56, 56, 512, 1, 2, 1, 0),
            (128, 128, 28, 28, 128, 3, 1, 1, 1),
            (128, 512, 28, 28, 128, 1, 1, 1, 0),
            (128, 512, 28, 28, 1024, 1, 2, 1, 0),
            (128, 256, 14, 14, 256, 3, 1, 1, 1),
            (128, 1024, 14, 14, 256, 1, 1, 1, 0),
            (128, 1024, 14, 14, 2048, 1, 2, 1, 0),
            (128, 512, 7, 7, 512, 3, 1, 1, 1),
            (128, 2048, 7, 7, 512, 1, 1, 1, 0),
            (128, 128, 56, 56, 128, 3, 2, 1, 1),
            (128, 3, 224, 224, 64, 7, 2, 1, 3),
            ],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=["triton"],
        line_names=["Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="conv2d-performance-fp16",  # Name for the plot, used also as a file name for saving the plot.
        args={"fp8_inputs": fp8_inputs},
    ))


@triton.testing.perf_report(configs)
def benchmark(N, C, H, W, F, K, S, D, P, provider, fp8_inputs):
    warmup = 25
    rep = 100
    a = torch.randint(low=-2, high=3, size=(N, H, W, C), device='cuda', dtype=torch.float16)
    b = torch.randint(low=-2, high=3, size=(K, K, C, F), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: conv2d(a, b, S, D, P), warmup=warmup, rep=rep, quantiles=quantiles)
    print(f"Time: {ms} ms")
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    perf = lambda ms: 2 * N * C * OH * OW * F * K * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=False, print_data=True)