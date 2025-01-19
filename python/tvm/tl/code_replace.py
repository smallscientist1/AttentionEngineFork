import re
import tvm
# from tvm.experiments.dequant_fp4_gemm import replace
# from tvm.experiments.mamba2_chunk_scan import replace
# from tvm.experiments.mamba2_chunk_state import replace

@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    # with open("main1.cu", "r", encoding='utf-8') as f:
    #     code = f.read()
    # code = replace(code)
    with open("main.cu", "w" ) as f:
        f.write(code)
    return code
