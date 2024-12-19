from .core import SymbolScalar, SymbolicArray, CustomIO
from .graph import Var, Const
from .utils import IndentedCode
from .tl_gen import generate_tl_from_dag
from .linear_attn_template import TlLinearAttnTemplate
import copy

from dataclasses import dataclass

shape_idx_map_o = {
    "batch": "bb",
    "heads": "bh",
    "seq_len": "by*BT:(by+1)*BT",
    "dimqk": "ik*BK:(ik+1)*BK",
    "dimv": "bx*BV:(bx+1)*BV",
    "1": "0"
    # others: ":" -> ":"
}
shape_idx_onchip_map_o = {
    "batch": "",
    "heads": "",
    "seq_len": "BT",
    "dimqk": "BK",
    "dimv": "BV",
    "1": ""
}

shape_idx_map_h = {
    "batch": "bb",
    "heads": "bhead",
    "seq_len": "i_t*BT:(i_t+1)*BT",
    "dimqk": "bx*BK:(bx+1)*BK",
    "dimv": "by*BV:(by+1)*BV",
    "1": "0"
    # others: ":" -> ":"
}
shape_idx_onchip_map_h = {
    "batch": "",
    "heads": "",
    "seq_len": "BT",
    "dimqk": "BK",
    "dimv": "BV",
    "1": ""
}
    
    
@dataclass
class lowerOutput:
    k_mod_expr: str = ""
    v_mod_expr: str = ""
    decay_mod_expr: str = ""
    q_mod_expr: str = ""
    custom_inputs_list: str = ""
    chunk_h_custom_inputs_list: str = ""
    k_mod_expr_fused_h: str = ""
    chunk_o_custom_inputs_list: str = ""
    v_mod_expr_fused_o: str = ""
    output_idx_list_h: str = "[3,]"
    output_idx_list_o: str = "[5,]"
    h_alloc_buffer_list: str = ""
    o_alloc_buffer_list: str = ""
    custom_inputs_list_o: str = ""
    custom_inputs_list_h: str = ""

@dataclass
class TunnerOutput:
    BT: str = "64"
    BK_h: str = "64"
    BV_h: str = "64"
    num_stages_h: str = "2"
    num_threads_h: str = "128"
    BK_o: str = "64"
    BV_o: str = "64"
    num_stages_o: str = "2"
    num_threads_o: str = "128"

def lowerKmod(k_mod, custom_io, lower_output: lowerOutput):
    k = SymbolicArray("k", Var("k"), shape_idx=["B", "H", "T", "D"])
    new_k = k_mod(k, custom_io)
    pytorch_code, input_vars = generate_tl_from_dag([new_k], to_tl=False)
    lower_output.k_mod_expr = str(pytorch_code)

    # custom_inputs_list = ", ".join([f"{varname}={varname}" for varname in input_vars.keys()])
    # custom_inputs_list += ","

def lowerVmod(v_mod, custom_io, lower_output: lowerOutput):
    v = SymbolicArray("v", Var("v"), shape_idx=["B", "H", "T", "D"])
    new_v = v_mod(v, custom_io)
    pytorch_code, input_vars = generate_tl_from_dag([new_v], to_tl=False)
    lower_output.v_mod_expr = str(pytorch_code)

# TODO: tmp solution
def lowerFusedVmod(v_mod, custom_io, lower_output: lowerOutput):
    vv = SymbolicArray("bs", Var("bs"), shape_idx=["BT", "BT"])
    custom_io1 = copy.deepcopy(custom_io)
    new_v = v_mod(vv, custom_io1)
    
    tl_code, input_vars = generate_tl_from_dag([new_v])
    input_vars.pop("bs")
    
    new_custom_io = CustomIO()
    for k, v in input_vars.items():
        new_custom_io.input_tensors[k] = copy.deepcopy(v)
        new_custom_io.input_tensors[k].clear_codegen()
        new_custom_io.input_tensors[k].varname = f"{k}_local"
        if len(v.shape_idx) == 4:
            raise("Not support shape_idx with 4")
        # else:
        #     lower_output.v_mod_expr_fused_o += \
        #     f"T.copy({k}[{','.join([shape_idx_map_o[i] for i in v.shape_idx])}], {k}_shared)\n"
        #     lower_output.v_mod_expr_fused_o += \
        #     f"T.copy({k}_shared, {k}_local)\n"
        #     new_custom_io.input_tensors[k].shape_idx = [shape_idx_onchip_map_o[ sidx] for sidx in v.shape_idx if shape_idx_onchip_map_o[ sidx]!=""]
        #     lower_output.o_alloc_buffer_list += f"{k}_shared = T.alloc_shared(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=dtype, scope='shared')\n"
        #     lower_output.o_alloc_buffer_list += f"{k}_local = T.alloc_fragment(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=accum_dtype)\n"
        elif v.shape_idx==["batch", "heads", "seq_len"]:
            new_custom_io.input_tensors[k].shape_idx = ["1", "BT"]
            lower_output.o_alloc_buffer_list += f"{k}_shared = T.alloc_shared(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=dtype, scope='shared')\n"
            lower_output.o_alloc_buffer_list += f"{k}_local = T.alloc_fragment(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=accum_dtype)\n"
            lower_output.v_mod_expr_fused_o += \
            f"T.copy({k}[{','.join([shape_idx_map_o[i] for i in v.shape_idx])}], {k}_shared)\n"
            lower_output.v_mod_expr_fused_o += \
            f"T.copy({k}_shared, {k}_local)\n"
            lower_output.chunk_o_custom_inputs_list += f"{k}: T.Buffer(({','.join(v.shape_idx)}), dtype),\n"
            lower_output.custom_inputs_list_o += f"{k},"
        else:
            raise("Not support shape_idx")
        
    lower_output.output_idx_list_o = f"[{len(input_vars)+5},]"    
    vv = SymbolicArray("bs", Var("bs"), shape_idx=["BT", "BT"])
    new_v = v_mod(vv, new_custom_io)
    tl_code, input_vars = generate_tl_from_dag([new_v])
    lower_output.v_mod_expr_fused_o += str(tl_code)
    
    ############################
    
    vv = SymbolicArray("b_k", Var("b_k"), shape_idx=["BT", "BK"])
    custom_io1 = copy.deepcopy(custom_io)
    new_v = v_mod(vv, custom_io1)
    
    tl_code, input_vars = generate_tl_from_dag([new_v])
    input_vars.pop("b_k")
    
    new_custom_io = CustomIO()
    for k, v in input_vars.items():
        new_custom_io.input_tensors[k] = copy.deepcopy(v)
        new_custom_io.input_tensors[k].clear_codegen()
        new_custom_io.input_tensors[k].varname = f"{k}_local"
        if len(v.shape_idx) == 4:
            raise("Not support shape_idx with 4")
        elif v.shape_idx==["batch", "heads", "seq_len"]:
            new_custom_io.input_tensors[k].shape_idx = ["BT"]
            lower_output.h_alloc_buffer_list += f"{k}_shared = T.alloc_shared(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=dtype, scope='shared')\n"
            lower_output.h_alloc_buffer_list += f"{k}_local = T.alloc_fragment(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=accum_dtype)\n"
            lower_output.k_mod_expr_fused_h += \
            f"T.copy({k}[{','.join([shape_idx_map_h[i] for i in v.shape_idx])}], {k}_shared)\n"
            lower_output.k_mod_expr_fused_h += \
            f"T.copy({k}_shared, {k}_local)\n"
            lower_output.chunk_h_custom_inputs_list += f"{k}: T.Buffer(({','.join(v.shape_idx)}), dtype),\n"
            lower_output.custom_inputs_list_h += f"{k},"
        else:
            raise("Not support shape_idx")
        
    lower_output.output_idx_list_h = f"[{len(input_vars)+3},]"    
    vv = SymbolicArray("b_k", Var("b_k"), shape_idx=["BT", "BK"])
    new_v = v_mod(vv, new_custom_io)
    tl_code, input_vars = generate_tl_from_dag([new_v])
    lower_output.k_mod_expr_fused_h += str(tl_code)
    
    

def lowerDecaymod(decay_mod, custom_io, lower_output: lowerOutput):
    decay = SymbolicArray("decay", Var("decay"), shape_idx=["B", "H", "T"])
    new_decay = decay_mod(decay, custom_io)
    pytorch_code, input_vars = generate_tl_from_dag([new_decay], to_tl=False)
    lower_output.decay_mod_expr = str(pytorch_code)

def lowerQmod(q_mod, custom_io, lower_output: lowerOutput):
    bq = SymbolScalar("bq", Var("bq"), shape_idx=["BT", "BK"])
    new_q = q_mod(bq, custom_io)
    tl_code, input_vars = generate_tl_from_dag([new_q])
    lower_output.q_mod_expr = str(tl_code)

def lower_tl(q_mod, k_mod, v_mod, decay_mod, custom_io, tuned_config=None):
    
    tune_output = TunnerOutput() if tuned_config is None else TunnerOutput(**tuned_config)
    
    lower_output = lowerOutput()
    if k_mod:
        lowerKmod(k_mod, custom_io, lower_output)
    if v_mod:
        try:
            lowerFusedVmod(v_mod, custom_io, lower_output)
        except:
            lowerVmod(v_mod, custom_io, lower_output)
    if decay_mod:
        lowerDecaymod(decay_mod, custom_io, lower_output)
    if q_mod:
        lowerQmod(q_mod, custom_io, lower_output)
    lower_output.custom_inputs_list = ", ".join([f"{varname}" for varname in custom_io.input_tensors.keys()])
    lower_output.custom_inputs_list += "," if lower_output.custom_inputs_list else ""
    return TlLinearAttnTemplate(
        **(lower_output.__dict__),
        **(tune_output.__dict__)
    )()


