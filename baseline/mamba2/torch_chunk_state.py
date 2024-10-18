import os
import sys
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import *

import torch

def chunk_state_ref(B, x, dt, dA_cumsum):
    from einops import rearrange, repeat
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)


def run_torch(batch, nheads, ngroups, seqlen, headdim, dstate, chunk_size, dtype=torch.float16):
    device = torch.device("cuda")

    n_chunks = seqlen // chunk_size
    flops = 2 * batch * seqlen * nheads * headdim * dstate

    B = torch.empty((batch, seqlen, ngroups, headdim), device=device, dtype=dtype).normal_(-1.0, 1.0)
    x = torch.empty((batch, seqlen, nheads, headdim), device=device, dtype=dtype).normal_(-1.0, 1.0)
    dt = torch.empty((batch, nheads, n_chunks, chunk_size), device=device, dtype=dtype).normal_(-1.0, 1.0)
    dA_cumsum = torch.empty((batch, nheads, n_chunks, chunk_size), device=device, dtype=dtype).normal_(-1.0, 1.0)

    compiled_func = torch.compile(chunk_state_ref)
    tflops, avg_latency = run_profile(compiled_func, B, x, dt, dA_cumsum, total_flops=flops)
    return tflops, avg_latency
    # print(f"TFLOPS: {tflops:.2f}")
    # print(f"Avg Latency: {avg_latency:.2f} ms")

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for model configuration")

    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--nheads', type=int, default=80, help='Number of attention heads')
    parser.add_argument('--ngroups', type=int, default=1, help='Number of groups')
    parser.add_argument('--seqlen', type=int, default=8192, help='Sequence length')
    parser.add_argument('--headdim', type=int, default=64, help='Head dimension size')
    parser.add_argument('--dstate', type=int, default=128, help='State dimension size')
    parser.add_argument('--chunk_size', type=int, default=256, help='Chunk size for the model')
    parser.add_argument('--dtype', type=str, default='float16', help='Data type for the model (e.g., float16)')

    args = parser.parse_args()
    assert args.dtype in ['float16', 'float32'], f"Unknown dtype: {args.dtype}"
    if args.dtype == 'float16':
        args.dtype = torch.float16
    elif args.dtype == 'float32':
        args.dtype = torch.float32

    return args

if __name__ == "__main__":
    args = parse_args()
    run_torch(args.batch, args.nheads, args.ngroups, args.seqlen, args.headdim, args.dstate, args.chunk_size, args.dtype)