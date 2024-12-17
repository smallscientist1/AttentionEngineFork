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
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
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
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=2),
        # # Good config for fp8 inputs.
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4)
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
def conv2d_pad_kernel(
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
    # inh = (h-1)*s +(kh-1) * d +1 -2*p
    # (oh-1)*s=inh + 2*p -1 - (kh-1)*d
    # 对的对的，上下一样
    # actually, should pad first then compute!

    # inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    # inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    # padh = inh + 2 * p
    # padw = inw + 2 * p
    # ir = f"pad[N, H0, W0, C] = input0[N, H0-{p}, W0-{p}, C].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, C, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
    #        data[N, K] = pad[N//{h*w}, N%{h*w}//{w}*{s}+K//{kw*c}*{d}, N%{w}*{s}+K//{c}%{kw}*{d}, K%{c}] where K in {kh*kw*c}, N in {n*h*w}; \
    #        output0[N, M] +=! data[N, K] * input1[K, M];"
    # input_dict = {
    #     "input0": {"dtype": dtype, "shape": [n, inh, inw, c]},
    #     "input1": {"dtype": dtype, "shape": [kh*kw*c, f]}
    # }

    PAD_H=H+2*P
    PAD_W=W+2*P

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

    # stride_am, stride_ak,
    # stride_bk, stride_bn,  #
    # stride_cm, stride_cn,

    # a.stride(0), a.stride(1) 256 1
    # b.stride(0), b.stride(1) 64 1
    # c.stride(0), c.stride(1) 64 1
    # this is for vectorized memory access in cuda gpus
    
    # this is actually from the view of output tensor, not input 
    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    # offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) 
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # output index
    n_idx=offs_cm // (OH*OW)
    # f_idx=off_cn 
    oh_idx=offs_cm % (OH * OW) //OW
    ow_idx=offs_cm % OW

    # all vector
    offs_am=n_idx*(PAD_H*PAD_W) + oh_idx*S*PAD_W + ow_idx*S
    
    offs_bn=offs_cn

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # # a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    aptr_base = a_ptr + (offs_am[:, None] * stride_am) + (offs_k[None, :] * stride_ak)

    aptr_base_org = a_ptr + (offs_cm[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(KH * KW * C, BLOCK_SIZE_K)):
        # c_index=(offs_k * BLOCK_SIZE_K) % C
        # kw_idex=((offs_k * BLOCK_SIZE_K)//C)%KW
        # kh_index=(offs_k * BLOCK_SIZE_K)//(C*KW)

        c_index=offs_k % C
        kw_idex=(offs_k//C)%KW
        kh_index=offs_k//(C*KW)

        # offs_ir_am=offs_am+kw_idex+kh_index*PAD_W

        # a_ptrs = a_ptr + (offs_am[:,None] * stride_am + c_index[None, :] * stride_ak)
        # a_ptrs = a_ptr + (offs_cm[:,None] * stride_am + c_index[None, :] * stride_ak)

        # a_ptrs = a_ptr + (offs_cm[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        # a_ptrs = a_ptr + (offs_cm[:, None] * stride_am) + (c_index[None, :] * stride_ak)

        # a_ptrs = a_ptr + (offs_am[:,None] * stride_am + c_index[None, :] * stride_ak + kw_idex[None,:] * stride_ak + kh_index[None,:]*stride_am)
        # a_ptrs = a_ptr + (offs_am[:,None] * stride_am + c_index[None, :] * stride_ak + kw_idex[None,:] * stride_ak + kh_index[None,:]*PAD_W*stride_am)
        a_ptrs = a_ptr + ((offs_am[:,None] + kw_idex[None,:] * D +kh_index[None,:]*PAD_W*D) * stride_am + c_index[None, :] * stride_ak )
        
        # aptrs = aptr_base + c_index + 

        # m = offs_cm
        # p = m % (OH * OW) // OW
        # q = m % OW
        # r = (k * BLOCK_SIZE_K) // (KW * C)
        # s = ((k * BLOCK_SIZE_K) // C) % KW
        # h = p[:, None] * S + r[None, :] * D - P
        # w = q[:, None] * S + s[None, :] * D - P
        # mask_x = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        # a_ptrs = aptr_base_org + ((k * BLOCK_SIZE_K) % C)
        a = tl.load(a_ptrs, mask=offs_k[None , :] < KH * KW * C , other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < KH * KW * C , other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        b_ptrs += BLOCK_SIZE_K * stride_bk

        offs_k += BLOCK_SIZE_K
    c = accumulator.to(tl.float16)
    # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < N * OH * OW) & (offs_cn[None, :] < F)
    tl.store(c_ptrs, c, mask=c_mask)



# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['N', 'C', 'H', 'W', 'F', 'K', 'S', 'D', 'P'],
# )
# @triton.jit
# def conv2d_no_pad_kernel(
#         # Pointers to matrices
#         a_ptr, b_ptr, c_ptr,
#         # Matrix dimensions
#         N, C, H, W, F, K, S, D, P,
#         # The stride variables represent how much to increase the ptr by when moving by 1
#         # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
#         # by to get the element one row down (A has M rows).
#         # stride_an, stride_ac, stride_ah, stride_aw, #
#         stride_am, stride_ak,
#         stride_bk, stride_bn,  #
#         stride_cm, stride_cn,
#         # Meta-parameters
#         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
#         GROUP_SIZE_M: tl.constexpr,  #
# ):
#     KH, KW = K, K

#     PAD_H=H+2*P
#     PAD_W=W+2*P

#     OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
#     OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(N * OH * OW, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(F, BLOCK_SIZE_N)
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n
    
#     # this is actually from the view of output tensor, not input 
#     # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
#     # offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

#     offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
#     offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) 
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     # a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     aptr_base = a_ptr + (offs_am[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
#     for k in range(0, tl.cdiv(KH * KW * C, BLOCK_SIZE_K)):
#         m = offs_am
#         p = m % (OH * OW) // OW
#         q = m % OW
#         r = (k * BLOCK_SIZE_K) // (KW * C)
#         s = ((k * BLOCK_SIZE_K) // C) % KW
#         h = p[:, None] * S + r[None, :] * D - P
#         w = q[:, None] * S + s[None, :] * D - P
#         mask_x = (h >= 0) & (h < H) & (w >= 0) & (w < W)
#         a_ptrs = aptr_base + ((k * BLOCK_SIZE_K) % C)
#         a = tl.load(a_ptrs, mask=mask_x, other=0.0)
#         b = tl.load(b_ptrs, mask=offs_k[:, None] < KH * KW * C - k * BLOCK_SIZE_K, other=0.0)
#         accumulator = tl.dot(a, b, accumulator)
#         b_ptrs += BLOCK_SIZE_K * stride_bk
#     c = accumulator.to(tl.float16)
#     # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < N * OH * OW) & (offs_cn[None, :] < F)
#     tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

# @triton.jit
# def padding(x):
#     return padded result
#     pass

# def conv2d(a, b, S, D, P):
#     N, H, W, C = a.shape
#     KH, KW, C, F = b.shape
#     assert KH == KW
#     K = KH
#     a = a.view(N * H * W, C)
#     b = b.view(KH * KW * C, F)
#     OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
#     OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
#     c = torch.empty((N * OH * OW, F), device=a.device, dtype=torch.float16)
#     grid = lambda META: (triton.cdiv(F, META['BLOCK_SIZE_N']) * triton.cdiv(N * OH * OW, META['BLOCK_SIZE_M']), )
#     print("a.stride(0), a.stride(1)",a.stride(0), a.stride(1))
#     print("b.stride(0), b.stride(1)",b.stride(0), b.stride(1))
#     print("c.stride(0), c.stride(1)",c.stride(0), c.stride(1))
#     conv2d_kernel[grid](
#         a, b, c,  #
#         N, C, H, W, F, K, S, D, P,
#         a.stride(0), a.stride(1),  #
#         b.stride(0), b.stride(1),  #
#         c.stride(0), c.stride(1),  #
#     )
#     return c.view(N, OH, OW, F)

def conv2d_pad(a, b, S, D, P):
    N, H, W, C = a.shape
    print("before padding a.shape", a.shape)
    a = torch.nn.functional.pad(a, (0 ,0 , P, P, P, P),value=0)
    N, pad_h, pad_w, c = a.shape
    print("after padding a.shape", a.shape)
    print("b.shape",b.shape)

    KH, KW, C, F = b.shape
    assert KH == KW
    K = KH
    a = a.view(N * pad_h * pad_w, C)
    b = b.view(KH * KW * C, F)
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
    c = torch.empty((N * OH * OW, F), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(F, META['BLOCK_SIZE_N']) * triton.cdiv(N * OH * OW, META['BLOCK_SIZE_M']), )
    print("a.stride(0), a.stride(1)",a.stride(0), a.stride(1))
    print("b.stride(0), b.stride(1)",b.stride(0), b.stride(1))
    print("c.stride(0), c.stride(1)",c.stride(0), c.stride(1))
    conv2d_pad_kernel[grid](
        a, b, c,  #
        N, C, H, W, F, K, S, D, P,
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    print("c.view(N, OH, OW, F).shape",c.view(N, OH, OW, F).shape)
    print("c.shape",c.shape)
    return c.view(N, OH, OW, F)

def ref_program(A, B, stride, padding, dilation):
    A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
    B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
    C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
    C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
    return C

torch.manual_seed(0)
# N, C, H, W, F, K, S, D, P = 1, 256, 7, 7, 256, 1, 1, 1, 0
# N, C, H, W, F, K, S, D, P = 1, 256, 7, 7, 256, 3, 1, 1, 1
# N, C, H, W, F, K, S, D, P = 1, 1, 8, 8, 1, 3, 1, 1, 1
# N, C, H, W, F, K, S, D, P = 128,256,56,56,64,1,1,1,0
# N, C, H, W, F, K, S, D, P = 128,256,56,56,64,1,1,1,0
# N, C, H, W, F, K, S, D, P = 128,256,56,56,64,1,1,1,0
# N, C, H, W, F, K, S, D, P = 128,256,56,56,64,1,2,1,1
# N, C, H, W, F, K, S, D, P = 128,256,56,56,64,1,3,1,1
# N, C, H, W, F, K, S, D, P = 128,256,56,56,64,3,2,1,1
# N, C, H, W, F, K, S, D, P = 128,256,56,56,64,3,2,2,1
N, C, H, W, F, K, S, D, P = 128, 2048, 7, 7, 512, 1, 1, 1, 0
# N, C, H, W, F, K, S, D, P = 128,256,56,56,64,3,1,1,0
# N, C, H, W, F, K, S, D, P = 1, 16, 7, 7, 16, 1, 1, 1, 0
# a = torch.randint(low=-2, high=3, size=(N, H, W, C), device='cuda', dtype=torch.float16)
# b = torch.randint(low=-2, high=3, size=(K, K, C, F), device='cuda', dtype=torch.float16)
# a = torch.ones(N, H, W, C, device='cuda', dtype=torch.float16)
# b = torch.ones(K, K, C, F, device='cuda', dtype=torch.float16)
a = torch.randint(low=-2, high=3, size=(N, H, W, C), device='cuda', dtype=torch.float16)
b = torch.randint(low=-2, high=3, size=(K, K, C, F), device='cuda', dtype=torch.float16)
triton_output = conv2d_pad(a, b, S, D, P)
torch_output = ref_program(a, b, stride=S, padding=P, dilation=D)
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    # count how many percent not match within 1e-2
    diff = torch.abs(triton_output - torch_output)
    not_match = torch.count_nonzero(diff > 1e-2)
    total = torch.numel(diff)
    print(f"❌ Triton and Torch differ by {not_match}/{total} ({not_match/total*100:.2f}%)")
    # print("triton_output:", triton_output)
    # print("torch_output:", torch_output)

fp8_inputs = False
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N", "C", "H", "W", "F", "K", "S", "D", "P"],  # Argument names to use as an x-axis for the plot
        x_vals=[
            (128, 2048, 7, 7, 512, 1, 1, 1, 0)
            # (1, 512, 7, 7, 2048, 1, 1, 1, 0),
            # (1, 512, 14, 14, 512, 3, 2, 1, 1),
            # (1, 1024, 14, 14, 512, 1, 1, 1, 0),
            # (1, 256, 14, 14, 1024, 1, 1, 1, 0),
            # (1, 256, 28, 28, 256, 3, 2, 1, 1),
            # (1, 512, 28, 28, 256, 1, 1, 1, 0),
            # (1, 128, 28, 28, 512, 1, 1, 1, 0),
            # (1, 256, 56, 56, 128, 1, 1, 1, 0),
            # (1, 64, 56, 56, 256, 1, 1, 1, 0),
            # (1, 64, 56, 56, 64, 3, 1, 1, 1),
            # (1, 64, 56, 56, 64, 1, 1, 1, 0),
            # (1, 256, 56, 56, 64, 1, 1, 1, 0),
            # (1, 256, 56, 56, 512, 1, 2, 1, 0),
            # (1, 128, 28, 28, 128, 3, 1, 1, 1),
            # (1, 512, 28, 28, 128, 1, 1, 1, 0),
            # (1, 512, 28, 28, 1024, 1, 2, 1, 0),
            # (1, 256, 14, 14, 256, 3, 1, 1, 1),
            # (1, 1024, 14, 14, 256, 1, 1, 1, 0),
            # (1, 1024, 14, 14, 2048, 1, 2, 1, 0),
            # (1, 512, 7, 7, 512, 3, 1, 1, 1),
            # (1, 2048, 7, 7, 512, 1, 1, 1, 0),
            # (1, 128, 56, 56, 128, 3, 2, 1, 1),
            # (1, 3, 224, 224, 64, 7, 2, 1, 3),
            # (128, 512, 7, 7, 2048, 1, 1, 1, 0),
            # (128, 512, 14, 14, 512, 3, 2, 1, 1),
            # (128, 1024, 14, 14, 512, 1, 1, 1, 0),
            # (128, 256, 14, 14, 1024, 1, 1, 1, 0),
            # (128, 256, 28, 28, 256, 3, 2, 1, 1),
            # (128, 512, 28, 28, 256, 1, 1, 1, 0),
            # (128, 128, 28, 28, 512, 1, 1, 1, 0),
            # (128, 256, 56, 56, 128, 1, 1, 1, 0),
            # (128, 64, 56, 56, 256, 1, 1, 1, 0),
            # (128, 64, 56, 56, 64, 3, 1, 1, 1),
            # (128, 64, 56, 56, 64, 1, 1, 1, 0),
            # (128, 256, 56, 56, 64, 1, 1, 1, 0),
            # (128, 256, 56, 56, 512, 1, 2, 1, 0),
            # (128, 128, 28, 28, 128, 3, 1, 1, 1),
            # (128, 512, 28, 28, 128, 1, 1, 1, 0),
            # (128, 512, 28, 28, 1024, 1, 2, 1, 0),
            # (128, 256, 14, 14, 256, 3, 1, 1, 1),
            # (128, 1024, 14, 14, 256, 1, 1, 1, 0),
            # (128, 1024, 14, 14, 2048, 1, 2, 1, 0),
            # (128, 512, 7, 7, 512, 3, 1, 1, 1),
            # (128, 2048, 7, 7, 512, 1, 1, 1, 0),
            # (128, 128, 56, 56, 128, 3, 2, 1, 1),
            # (128, 3, 224, 224, 64, 7, 2, 1, 3),
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
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: conv2d_pad(a, b, S, D, P), warmup=warmup, rep=rep, quantiles=quantiles)
    print(f"Time: {ms} ms")
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    perf = lambda ms: 2 * N * C * OH * OW * F * K * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=False, print_data=False)