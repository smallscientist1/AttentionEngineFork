import torch

def run_profile(func, *ins, total_flops=0, _warmups=10, _runs=10):
    # print(f"Running {func.__name__} with inputs {ins}")
    warmups = _warmups
    runs = _runs
    for _ in range(warmups):
        out = func(*ins)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for _ in range(runs):
        out = func(*ins)

    torch.cuda.synchronize()
    end_event.record()

    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event)
    avg_latency = latency / runs

    tflops = total_flops / latency * runs * 1e-9
    return tflops, avg_latency