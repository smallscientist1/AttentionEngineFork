import os
import sys
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import *

import torch

def chunk_scan_ref(cb, x, dt, dA_cumsum, C, prev_states):
    from einops import rearrange, repeat
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, headdim = x.shape
    # _, _, ngroups, dstate = B.shape
    # assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    # assert C.shape == B.shape
    # B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)
    # CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
    #                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")

    return out

def run_torch(batch, nheads, ngroups, seqlen, headdim, dstate, chunk_size, dtype=torch.float16):
    device = torch.device("cuda")

    n_chunks = seqlen // chunk_size
    flops = 2.0 * batch * seqlen * chunk_size * nheads * headdim * 0.5 + 2.0 * batch * seqlen * nheads * headdim * dstate

    cb = torch.empty((batch, n_chunks, ngroups, chunk_size, chunk_size), device=device, dtype=dtype).normal_(-1.0, 1.0)
    x = torch.empty((batch, seqlen, nheads, headdim), device=device, dtype=dtype).normal_(-1.0, 1.0)
    dt = torch.empty((batch, nheads, n_chunks, chunk_size), device=device, dtype=dtype).normal_(-1.0, 1.0)
    dA_cumsum = torch.empty((batch, nheads, n_chunks, chunk_size), device=device, dtype=dtype).normal_(-1.0, 1.0)
    c = torch.empty((batch, seqlen, ngroups, dstate), device=device, dtype=dtype).normal_(-1.0, 1.0)
    prev_states = torch.empty((batch, n_chunks, nheads, headdim, dstate), device=device, dtype=dtype).normal_(-1.0, 1.0)

    compiled_func = torch.compile(chunk_scan_ref)
    tflops, avg_latency = run_profile(compiled_func, cb, x, dt, dA_cumsum, c, prev_states, total_flops=flops)
    return tflops, avg_latency
    print(f"TFLOPS: {tflops:.2f}")
    print(f"Avg Latency: {avg_latency:.2f} ms")

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