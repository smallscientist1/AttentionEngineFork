import bitblas
import torch

# uncomment to enable debug output
# bitblas.set_log_level("Debug")


def run(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, check=True):
    matmul_config = bitblas.MatmulConfig(
        M=M,  # M dimension
        N=N,  # N dimension
        K=K,  # K dimension
        A_dtype=A_dtype,  # activation A dtype
        W_dtype=W_dtype,  # weight W dtype
        accum_dtype=accum_dtype,  # accumulation dtype
        out_dtype=out_dtype,  # output dtype
        layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
        with_bias=False,  # bias
        # configs for weight only quantization
        group_size=None,  # setting for grouped quantization
        with_scaling=False,  # setting for scaling factor
        with_zeros=False,  # setting for zeros
        zeros_mode=None,  # setting for how to calculating zeros
    )

    matmul = bitblas.Matmul(config=matmul_config)

    # Create input matrices
    input_tensor = torch.rand((M, K), dtype=torch.float16).cuda()
    weight_tensor = torch.randint(0, 7, (N, K), dtype=torch.int8).cuda()

    # Transform weight tensor to int4 data type
    weight_tensor_transformed = matmul.transform_weight(weight_tensor)

    # Perform mixed-precision matrix multiplication
    output_tensor = matmul(input_tensor, weight_tensor_transformed)

    # Reference result using PyTorch matmul for comparison
    ref_result = torch.matmul(input_tensor, weight_tensor.t().to(torch.float16))
    # Assert that the results are close within a specified tolerance, note that the int4 randint value is a little bigger than the float16 value, so we set the atol to 1.0
    # print("Ref output:", ref_result)
    # print("BitBLAS output:", output_tensor)
    if check:
        torch.testing.assert_close(output_tensor, ref_result, rtol=1e-2, atol=1e-0)

    warmups = 10
    runs = 10
    for _ in range(warmups):
        out = matmul(input_tensor, weight_tensor_transformed)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for _ in range(runs):
        out = matmul(input_tensor, weight_tensor_transformed)

    torch.cuda.synchronize()
    end_event.record()

    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event)

    flops_per_matmul = 2.0 * M * N * K
    total_flops = 2 * flops_per_matmul
    tflops = total_flops / latency * runs * 1e-9
    print(f"TFLOPS: {tflops}")


if __name__ == "__main__":
    M, N, K = 8192, 8192, 8192
    A_dtype, W_dtype, accum_dtype, out_dtype = "float16", "int4", "float32", "float16"
    run(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, check=False)