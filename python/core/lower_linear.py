from .core import SymbolScalar, SymbolicArray, CustomIO
from .graph import Var, Const
from .utils import IndentedCode
from .tl_gen import generate_tl_from_dag
from .linear_attn_template import TlLinearAttnTemplate

from dataclasses import dataclass

@dataclass
class lowerOutput:
    k_mod_expr: str = ""
    v_mod_expr: str = ""
    decay_mod_expr: str = ""
    q_mod_expr: str = ""
    custom_inputs_list: str = ""

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


