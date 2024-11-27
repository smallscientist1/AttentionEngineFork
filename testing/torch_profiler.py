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

# shapes = [
#     (4096,1024,4096),
#     (4096,4096,4096),
#     (4096,14336,4096),
#     (4096,4096,14336),
#     (8192,1024,4096),
#     (8192,4096,4096),
#     (8192,14336,4096),
#     (8192,4096,14336),
#     (4096,1024,8192),
#     (4096,8192,8192),
#     (4096,28672,8192),
#     (4096,8192,28672),
#     (8192,1024,8192),
#     (8192,8192,8192),
#     (8192,28672,8192),
#     (8192,8192,28672),
# ]

# for m, k, n in shapes:
#     print(f"Running for shapes: {m}, {k}, {n}")
#     A = torch.randn(m, k, dtype=torch.float16, device=device)
#     B = torch.randn(k, n, dtype=torch.float16, device=device)
#     profile_function(gemm, A, B)

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