from .core import SymbolScalar, SymbolicArray, CustomIO
from .graph import Var, Const
from .utils import IndentedCode
from .tl_gen import generate_tl_from_dag
from .linear_attn_template import TlLinearAttnTemplate

from dataclasses import dataclass

@dataclass
class lowerKmodOutput:
    k_mod_expr: str = ""
    # custom_inputs_list: str = ""

@dataclass
class lowerVmodOutput:
    v_mod_expr: str = ""

@dataclass
class lowerDecaymodOutput:
    decay_mod_expr: str = ""

@dataclass
class lowerQmodOutput:
    q_mod_expr: str = ""

def lowerKmod(k_mod, custom_io) -> lowerKmodOutput:
    k = SymbolicArray("k", Var("k"), shape_idx=["B", "H", "T", "D"])
    new_k = k_mod(k, custom_io)
    pytorch_code, input_vars = generate_tl_from_dag([new_k], to_tl=False)
    k_mod_expr = str(pytorch_code)

    # custom_inputs_list = ", ".join([f"{varname}={varname}" for varname in input_vars.keys()])
    # custom_inputs_list += ","
    return lowerKmodOutput(k_mod_expr=k_mod_expr)# , custom_inputs_list=custom_inputs_list)

def lowerVmod(v_mod, custom_io) -> lowerVmodOutput:
    v = SymbolicArray("v", Var("v"), shape_idx=["B", "H", "T", "D"])
    new_v = v_mod(v, custom_io)
    pytorch_code, input_vars = generate_tl_from_dag([new_v], to_tl=False)
    v_mod_expr = str(pytorch_code)
    return lowerVmodOutput(v_mod_expr=v_mod_expr)

def lowerDecaymod(decay_mod, custom_io) -> lowerDecaymodOutput:
    decay = SymbolicArray("decay", Var("decay"), shape_idx=["B", "H", "T"])
    new_decay = decay_mod(decay, custom_io)
    pytorch_code, input_vars = generate_tl_from_dag([new_decay], to_tl=False)
    decay_mod_expr = str(pytorch_code)
    return lowerDecaymodOutput(decay_mod_expr=decay_mod_expr)

def lowerQmod(q_mod, custom_io) -> lowerQmodOutput:
    bq = SymbolScalar("bq", Var("bq"), shape_idx=["BT", "BK"])
    new_q = q_mod(bq, custom_io)
    tl_code, input_vars = generate_tl_from_dag([new_q])
    q_mod_expr = str(tl_code)
    return lowerQmodOutput(q_mod_expr=q_mod_expr)

def lower_tl(q_mod, k_mod, v_mod, decay_mod, custom_io):
    if k_mod:
        lower_kmod_output = lowerKmod(k_mod, custom_io)
    if v_mod:
        lower_vmod_output = lowerVmod(v_mod, custom_io)
    if decay_mod:
        lower_decaymod_output = lowerDecaymod(decay_mod, custom_io)
    if q_mod:
        lower_qmod_output = lowerQmod(q_mod, custom_io)
    custom_inputs_list = ", ".join([f"{varname}" for varname in custom_io.input_tensors.keys()])
    custom_inputs_list += "," if custom_inputs_list else ""
    return TlLinearAttnTemplate(
        **(lower_kmod_output.__dict__ if k_mod else lowerKmodOutput().__dict__),
        **(lower_vmod_output.__dict__ if v_mod else lowerVmodOutput().__dict__),
        **(lower_decaymod_output.__dict__ if decay_mod else lowerDecaymodOutput().__dict__),
        **(lower_qmod_output.__dict__ if q_mod else lowerQmodOutput().__dict__),
        custom_inputs_list=custom_inputs_list
    )()


