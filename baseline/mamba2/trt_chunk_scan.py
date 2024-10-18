import numpy as np
import argparse

PREPARE_ONNX = False
warmups, iterations = 10, 10

  
def prepare_onnx(batch, nheads, ngroups, seqlen, headdim, dstate, chunk_size, dtype):
    import torch
    from torch import nn

    device = torch.device("cuda")
    n_chunks = seqlen // chunk_size
    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "float32":
        dtype = torch.float32
    
    class ChunkScanModel(nn.Module):
        def __init__(self):
            super(ChunkScanModel, self).__init__()

        def forward(self, cb, x, dt, dA_cumsum, C, prev_states):
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

    def export_onnx():
        cb = torch.empty((batch, n_chunks, ngroups, chunk_size, chunk_size), device=device, dtype=dtype).normal_(-1.0, 1.0)
        x = torch.empty((batch, seqlen, nheads, headdim), device=device, dtype=dtype).normal_(-1.0, 1.0)
        dt = torch.empty((batch, nheads, n_chunks, chunk_size), device=device, dtype=dtype).normal_(-1.0, 1.0)
        dA_cumsum = torch.empty((batch, nheads, n_chunks, chunk_size), device=device, dtype=dtype).normal_(-1.0, 1.0)
        C = torch.empty((batch, seqlen, ngroups, dstate), device=device, dtype=dtype).normal_(-1.0, 1.0)
        prev_states = torch.empty((batch, n_chunks, nheads, headdim, dstate), device=device, dtype=dtype).normal_(-1.0, 1.0)

        model = ChunkScanModel()

        torch.onnx.export(
            model,
            (cb, x, dt, dA_cumsum, C, prev_states),
            "chunk_scan_model_" + "_".join([str(x) for x in [batch, nheads, ngroups, seqlen, headdim, dstate, chunk_size]]) + ".onnx",
            input_names=['cb', 'x', 'dt', 'dA_cumsum', 'C', 'prev_states'],
            output_names=['out'],
            opset_version=14
        )

    export_onnx()

def build_engine(onnx_file_path):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX Parsing Error {error}: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse the ONNX model.")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 80 * (1 << 30))

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized engine.")

        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        return engine

def infer(engine, inputs, h_output):
    import pycuda.driver as cuda
    import pycuda.autoinit

    d_inputs = []
    for input_tensor in inputs:
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        cuda.memcpy_htod(d_input, input_tensor)
        d_inputs.append(d_input)

    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    start_event = cuda.Event()
    end_event = cuda.Event()

    with engine.create_execution_context() as context:
        bindings = [int(d_inp) for d_inp in d_inputs] + [int(d_output)]

        for _ in range(warmups):
            context.execute_v2(bindings=bindings)
            stream.synchronize()

        total_time = 0.0
        for _ in range(iterations):
            start_event.record(stream)
            context.execute_v2(bindings=bindings)
            end_event.record(stream)
            stream.synchronize()
            elapsed_time = start_event.time_till(end_event)
            total_time += elapsed_time
        
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    avg_latency = total_time / iterations
    return avg_latency

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

    return args

def run_trt(batch, nheads, ngroups, seqlen, headdim, dstate, chunk_size, dtype="float16"):
    assert dtype in ["float16", "float32"], f"Unknown dtype: {dtype}"
    if PREPARE_ONNX:
        prepare_onnx(batch, nheads, ngroups, seqlen, headdim, dstate, chunk_size, dtype)
        return 0, 0
    else:
        if dtype == "float16":
            np_dtype = np.float16
        elif dtype == "float32":
            np_dtype = np.float32

        n_chunks = seqlen // chunk_size
        flops = 2.0 * batch * seqlen * chunk_size * nheads * headdim * 0.5 + 2.0 * batch * seqlen * nheads * headdim * dstate

        cb = np.random.normal(-1.0, 1.0, (batch, n_chunks, ngroups, chunk_size, chunk_size)).astype(np_dtype)
        x = np.random.normal(-1.0, 1.0, (batch, seqlen, nheads, headdim)).astype(np_dtype)
        dt = np.random.normal(-1.0, 1.0, (batch, nheads, n_chunks, chunk_size)).astype(np_dtype)
        dA_cumsum = np.random.normal(-1.0, 1.0, (batch, nheads, n_chunks, chunk_size)).astype(np_dtype)
        C = np.random.normal(-1.0, 1.0, (batch, seqlen, ngroups, dstate)).astype(np_dtype)
        prev_states = np.random.normal(-1.0, 1.0, (batch, n_chunks, nheads, headdim, dstate)).astype(np_dtype)

        input_data = [cb, x, dt, dA_cumsum, C, prev_states]
        h_output = np.empty((batch, seqlen, nheads, headdim), dtype=np_dtype)

        onnx_file_path = "chunk_scan_model_" + "_".join([str(x) for x in [batch, nheads, ngroups, seqlen, headdim, dstate, chunk_size]]) + ".onnx"
        engine = build_engine(onnx_file_path)
        avg_latency = infer(engine, input_data, h_output)
        tflops = flops / avg_latency * 1e-9
        return tflops, avg_latency
        print(f"TFLOPS: {tflops:.2f}")
        print(f"Average latency: {avg_latency:.2f} ms")
        # print("Model Output:", output)

if __name__ == "__main__":
    args = parse_args()
    run_trt(args.batch, args.nheads, args.ngroups, args.seqlen, args.headdim, args.dstate, args.chunk_size, args.dtype)