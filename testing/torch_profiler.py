import torch
import time

def profile_function(fn, *args, **kwargs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.synchronize()
    else:
        device = torch.device("cpu")
    
    for _ in range(10):
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    start = time.time()
    for _ in range(repeats):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    end = time.time()

    avg_time = (end - start) / repeats * 1000
    print(f"Average execution time over {repeats} runs: {avg_time:.4f} ms")

def gemm(A, B):
    return A @ B

repeats = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

shapes = [
    # (1,14336,4096),
    # (1,4096,14336),
    # (32,1024,4096),
    # (32,4096,4096),
    # (32,14336,4096),
    # (32,4096,14336),
    # (64,1024,4096),
    # (64,4096,4096),
    # (64,14336,4096),
    # (64,4096,14336),
    # (128,1024,4096),
    # (128,4096,4096),
    # (128,14336,4096),
    # (128,4096,14336),
    # (1,28672,8192),
    # (1,8192,28672),
    # (32,1024,8192),
    # (32,8192,8192),
    # (32,28672,8192),
    # (32,8192,28672),
    # (64,1024,8192),
    # (64,8192,8192),
    # (64,28672,8192),
    # (64,8192,28672),
    # (128,1024,8192),
    # (128,8192,8192),
    # (128,28672,8192),
    # (128,8192,28672),
    (4096,1024,4096),
    (4096,4096,4096),
    (4096,14336,4096),
    (4096,4096,14336),
    (8192,1024,4096),
    (8192,4096,4096),
    (8192,14336,4096),
    (8192,4096,14336),
    (4096,1024,8192),
    (4096,8192,8192),
    (4096,28672,8192),
    (4096,8192,28672),
    (8192,1024,8192),
    (8192,8192,8192),
    (8192,28672,8192),
    (8192,8192,28672),
    # (32,1024,4096),
    # (32,4096, 4096),
    # (32, 14336, 4096),
    # (32, 4096, 14336),
    # (32,1024,8192),
    # (32,8192,8192),
    # (32,28672,8192),
    # (32,8192,28672),
]

for m, n, k in shapes:
    print(f"Running for shapes: {m}, {n}, {k}")
    A = torch.randn(m, k, dtype=torch.float16, device=device)
    B = torch.randn(k, n, dtype=torch.float16, device=device)
    profile_function(gemm, A, B)

shapes = [
    (1,512,7,7,2048,1,1,1,0),
    (1,512,14,14,512,3,2,1,1),
    (1,1024,14,14,512,1,1,1,0),
    (1,256,14,14,1024,1,1,1,0),
    (1,256,28,28,256,3,2,1,1),
    (1,512,28,28,256,1,1,1,0),
    (1,128,28,28,512,1,1,1,0),
    (1,256,56,56,128,1,1,1,0),
    (1,64,56,56,256,1,1,1,0),
    (1,64,56,56,64,3,1,1,1),
    (1,64,56,56,64,1,1,1,0),
    (1,256,56,56,64,1,1,1,0),
    (1,256,56,56,512,1,2,1,0),
    (1,128,28,28,128,3,1,1,1),
    (1,512,28,28,128,1,1,1,0),
    (1,512,28,28,1024,1,2,1,0),
    (1,256,14,14,256,3,1,1,1),
    (1,1024,14,14,256,1,1,1,0),
    (1,1024,14,14,2048,1,2,1,0),
    (1,512,7,7,512,3,1,1,1),
    (1,2048,7,7,512,1,1,1,0),
    (1,128,56,56,128,3,2,1,1),
    # (1,3,224,224,64,7,2,1,3),
    # (32,512,7,7,2048,1,1,1,0),
    # (32,512,14,14,512,3,2,1,1),
    # (32,1024,14,14,512,1,1,1,0),
    # (32,256,14,14,1024,1,1,1,0),
    # (32,256,28,28,256,3,2,1,1),
    # (32,512,28,28,256,1,1,1,0),
    # (32,128,28,28,512,1,1,1,0),
    # (32,256,56,56,128,1,1,1,0),
    # (32,64,56,56,256,1,1,1,0),
    # (32,64,56,56,64,3,1,1,1),
    # (32,64,56,56,64,1,1,1,0),
    # (32,256,56,56,64,1,1,1,0),
    # (32,256,56,56,512,1,2,1,0),
    # (32,128,28,28,128,3,1,1,1),
    # (32,512,28,28,128,1,1,1,0),
    # (32,512,28,28,1024,1,2,1,0),
    # (32,256,14,14,256,3,1,1,1),
    # (32,1024,14,14,256,1,1,1,0),
    # (32,1024,14,14,2048,1,2,1,0),
    # (32,512,7,7,512,3,1,1,1),
    # (32,2048,7,7,512,1,1,1,0),
    # (32,128,56,56,128,3,2,1,1),
    # (32,3,224,224,64,7,2,1,3),
    # (128,512,7,7,2048,1,1,1,0),
    # (128,512,14,14,512,3,2,1,1),
    # (128,1024,14,14,512,1,1,1,0),
    # (128,256,14,14,1024,1,1,1,0),
    # (128,256,28,28,256,3,2,1,1),
    # (128,512,28,28,256,1,1,1,0),
    # (128,128,28,28,512,1,1,1,0),
    # (128,256,56,56,128,1,1,1,0),
    # (128,64,56,56,256,1,1,1,0),
    # (128,64,56,56,64,3,1,1,1),
    # (128,64,56,56,64,1,1,1,0),
    # (128,256,56,56,64,1,1,1,0),
    # (128,256,56,56,512,1,2,1,0),
    # (128,128,28,28,128,3,1,1,1),
    # (128,512,28,28,128,1,1,1,0),
    # (128,512,28,28,1024,1,2,1,0),
    # (128,256,14,14,256,3,1,1,1),
    # (128,1024,14,14,256,1,1,1,0),
    # (128,1024,14,14,2048,1,2,1,0),
    # (128,512,7,7,512,3,1,1,1),
    # (128,2048,7,7,512,1,1,1,0),
    # (128,128,56,56,128,3,2,1,1),
    # (128,3,224,224,64,7,2,1,3),
]

def conv2d_nchw(A, B, stride, padding, dilation):
    C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
    return C

def conv2d_nhwc(A, B, stride, padding, dilation):
    A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
    B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
    C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
    C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
    return C

for N, C, H, W, F, K, S, D, P in shapes:
    # print(f"Running for shapes: {N}, {C}, {H}, {W}, {F}, {K}, {S}, {D}, {P}")
    # A = torch.randn(N, C, H, W, dtype=torch.float16, device=device)
    # B = torch.randn(F, C ,K, K, dtype=torch.float16, device=device)
    # profile_function(conv2d_nchw, A, B, S, P, D)

    A = torch.randn(N, H, W, C, dtype=torch.float16, device=device)
    B = torch.randn(K, K, C, F, dtype=torch.float16, device=device)
    profile_function(conv2d_nhwc, A, B, S, P, D)

def chunk_state_triton(B, x, dt, dA_cumsum):
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd
    return _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=False)

def chunk_scan_triton(cb, x, dt, dA_cumsum, C, states):
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
    out, _ =  _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states)
    return out

configs = [
    (1, 64, 1, 1024, 64, 128),
    (1, 64, 1, 2048, 64, 128),
    (1, 64, 1, 8192, 64, 128),
    (1, 64, 1, 16384, 64, 128),
    (64, 64, 1, 1024, 64, 128),
    (64, 64, 1, 2048, 64, 128),
    (64, 64, 1, 8192, 64, 128),
    (64, 64, 1, 16384, 64, 128),
    # (1, 80, 1, 1024, 64, 128),
    # (1, 80, 1, 2048, 64, 128),
    # (1, 80, 1, 8192, 64, 128),
    # (1, 80, 1, 16384, 64, 128),
    # (64, 80, 1, 1024, 64, 128),
    # (64, 80, 1, 2048, 64, 128),
    # (64, 80, 1, 8192, 64, 128),
    # (64, 80, 1, 16384, 64, 128),
]
chunk_size = 256

for config in configs:
    batch, nheads, ngroups, seqlen, headdim, dstate = config
    nchunks = seqlen // chunk_size
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float16, device=device)
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device=device)
    dt = torch.randn(batch, nheads, nchunks, chunk_size, dtype=torch.float16, device=device)
    dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size, dtype=torch.float16, device=device)
    profile_function(chunk_state_triton, B, x, dt, dA_cumsum)


    cb = torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, dtype=torch.float16, device=device)
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16, device=device)
    dt = torch.randn(batch, nheads, nchunks, chunk_size, dtype=torch.float16, device=device)
    dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size, dtype=torch.float16, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=torch.float16, device=device)
    prev_states = torch.randn(batch, nchunks, nheads, headdim, dstate, dtype=torch.float16, device=device)
    D = torch.randn(nheads, dtype=torch.float16, device=device)
    profile_function(chunk_scan_triton, cb, x, dt, dA_cumsum, C, prev_states)

configs = [
    (1, 32, 512, 128, True),
    (1, 32, 512, 128, False),
    (1, 32, 1024, 128, True),
    (1, 32, 1024, 128, False),
    (1, 64, 512, 128, True),
    (1, 64, 512, 128, False),
    (1, 64, 1024, 128, True),
    (1, 64, 1024, 128, False),
    # (1, 8, 512, 64, True),
    # (1, 8, 512, 64, False),
    # (64, 8, 512, 64, True),
    # (64, 8, 512, 64, False),
    # (1, 6, 1024, 64, True),
    # (1, 6, 1024, 64, False),
    # (64, 6, 1024, 64, True),
    # (64, 6, 1024, 64, False),
    # (1, 32, 4096, 128, True),
    # (1, 64, 4096, 128, True),
    # (1, 32, 4096, 128, False),
    # (1, 64, 4096, 128, False),
    # (1, 32, 8192, 128, True),
    # (1, 64, 8192, 128, True),
    # (1, 32, 8192, 128, False),
    # (1, 64, 8192, 128, False),
    # (64, 32, 4096, 128, True),
    # (64, 64, 4096, 128, True),
    # (64, 32, 4096, 128, False),
    # (64, 64, 4096, 128, False),
]

def torch_flash_attn(query_states, key_states, value_states, is_causal):
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    return attn_output


for config in configs:
    batch, nheads, seqlen, dim, causal = config
    Q = torch.randn(batch, nheads, seqlen, dim, dtype=torch.float16, device=device)
    K = torch.randn(batch, nheads, seqlen, dim, dtype=torch.float16, device=device)
    V = torch.randn(batch, nheads, seqlen, dim, dtype=torch.float16, device=device)

    profile_function(torch_flash_attn, Q, K, V, causal)
    total_tflops = 4 * batch * nheads * seqlen * seqlen * dim / 1e12
    
configs = [
    (1, 32, 1, 8192, 128, False),
    (1, 32, 128, 8192, 128, False),
    (1, 64, 1, 8192, 128, False),
    (1, 64, 128, 8192, 128, False),
]
for config in configs:
    batch, nheads, seqlen_q, seqlen_kv, dim, causal = config
    Q = torch.randn(batch, nheads, seqlen_q, dim, dtype=torch.float16, device=device)
    K = torch.randn(batch, nheads, seqlen_kv, dim, dtype=torch.float16, device=device)
    V = torch.randn(batch, nheads, seqlen_kv, dim, dtype=torch.float16, device=device)

    profile_function(torch_flash_attn, Q, K, V, causal)