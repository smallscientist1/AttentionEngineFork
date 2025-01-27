# from ..attn_engine import OnlineFunc
from .core import SymbolScalar, SymbolicArray, CustomIO
from .graph import Var, Const
from .utils import IndentedCode
from .tl_gen import generate_tl_from_dag
from .attn_template import TlAttnTemplate
from dataclasses import dataclass

# TODO: bwd map
shape_idx_map = {
    "batch": "bz",
    "heads": "by",
    "seq_len": "bx*block_M:(bx+1)*block_M",
    "seq_len_kv": "k*block_N:(k+1)*block_N",
    "1": "0"
    # others: ":" -> ":"
}
shape_idx_onchip_map = {
    "batch": "",
    "heads": "",
    "seq_len": "block_M",
    "seq_len_kv": "block_N",
    "1": ""
}
shape_idx_map_bwd = {
    "batch": "bz",
    "heads": "by",
    "seq_len": "k*block_N:(k+1)*block_N",
    "seq_len_kv": "bx*block_M:(bx+1)*block_M",
    "1": "0"
}
shape_idx_onchip_map_bwd = {
    "batch": "",
    "heads": "",
    "seq_len": "block_N",
    "seq_len_kv": "block_M",
    "1": ""
}

RECURRENT_DIM = "block_N"


@dataclass
class lowerOutput:
    swizzle_shared: str = ""
    tl_dtype: str = "float16"
    is_inf_mask: str = "True"


@dataclass
class TunnerOutput:
    block_M: str = "128"
    block_N: str = "128"
    stages: str = "2"
    thread_num: str = "256"
    shared_fuse: str = "False"


class lowerOnlineFuncOutput:

    def __init__(self, final_rowscales_output, final_rowscales_init, online_rowscales_initvalue, online_func_init, online_func_inputs, online_func_body, online_func_inputs_list, o_scale, online_func_epilogue, final_rowscales_save,
                 online_rowscales_update,
                 isused_doosum, final_rowscales_length, final_rowscales_load, online_func_fwd, custom_bwd_inputs_load, custom_bwd_body,
                 final_rowscales_shared_init,
                 custom_bwd_inputs, custom_bwd_inputs_init,
                 o_scale_varname):
        self.final_rowscales_output = final_rowscales_output
        self.final_rowscales_init = final_rowscales_init
        self.online_rowscales_initvalue = online_rowscales_initvalue
        self.online_func_init = online_func_init
        self.online_func_inputs = online_func_inputs
        self.online_func_body = online_func_body
        self.online_func_inputs_list = online_func_inputs_list
        self.o_scale = o_scale
        self.online_func_epilogue = online_func_epilogue
        self.final_rowscales_save = final_rowscales_save
        self.online_rowscales_update = online_rowscales_update

        self.isused_doosum = isused_doosum
        self.final_rowscales_length = final_rowscales_length
        self.final_rowscales_load = final_rowscales_load
        self.online_func_fwd = online_func_fwd
        self.custom_bwd_inputs_load = custom_bwd_inputs_load
        self.custom_bwd_body = custom_bwd_body
        self.final_rowscales_shared_init = final_rowscales_shared_init
        self.custom_bwd_inputs = custom_bwd_inputs
        self.custom_bwd_inputs_init = custom_bwd_inputs_init
        self.o_scale_varname = o_scale_varname


class lowerScoreModOutput:
    def __init__(self, score_mod_inputs, score_mod_body, score_mod_inputs_list,
                 score_mod_backward,
                 score_mod_bwd_inputs_list,
                 score_mod_bwd_inputs,
                 score_mod_inputs_bwd_list,
                 score_mod_fwd_inputs,
                 score_mod_fwd_body,
                 score_mod_output_var,
                 score_mod_bwd_inputs_declare,
                 score_mod_bwd_inputs_declare_shared
                 ):
        self.score_mod_inputs = score_mod_inputs
        self.score_mod_body = score_mod_body
        self.score_mod_inputs_list = score_mod_inputs_list
        self.score_mod_backward = score_mod_backward
        self.score_mod_bwd_inputs = score_mod_bwd_inputs
        self.score_mod_bwd_inputs_list = score_mod_bwd_inputs_list
        self.score_mod_inputs_bwd_list = score_mod_inputs_bwd_list
        self.score_mod_fwd_inputs = score_mod_fwd_inputs
        self.score_mod_fwd_body = score_mod_fwd_body
        self.score_mod_output_var = score_mod_output_var
        self.score_mod_bwd_inputs_declare = score_mod_bwd_inputs_declare
        self.score_mod_bwd_inputs_declare_shared = score_mod_bwd_inputs_declare_shared


def lower_online_func(online_func, lower_output: lowerOutput,
                      scores_name="scores"):  # OnlineFunc):
    online_fwd = online_func.online_fwd
    scores = SymbolicArray(
        scores_name,
        Var(scores_name),
        shape_idx=[
            "block_M",
            "block_N"])
    online_rowscales = online_func.online_rowscales
    b = SymbolScalar("b", Var("b"))
    h = SymbolScalar("h", Var("h"))
    q_idx = SymbolScalar("q_idx", Var("q_idx"))

    # online&epilogue
    input_vars = {}

    # generate init value for online_rowscales
    online_rowscales_initvalue = ""
    for k, v in online_rowscales.items():  # v.code is Var
        if v.code.name == "-inf":
            tl_init_value = "-T.infinity(accum_dtype)"
        else:
            tl_init_value = v.code.name
        online_rowscales_initvalue += f"T.fill({v.varname}, {tl_init_value})\n"

    # online_fwd
    scores_new, new_online_rowscales, o_scalevar = online_fwd(
        scores, online_rowscales, b, h, q_idx)
    for k, v in new_online_rowscales.items():
        new_online_rowscales[k].count += 1
    o_scalevar.count += 1
    tl_code, input_vars_online = generate_tl_from_dag(
        list(new_online_rowscales.values()) + [scores_new, o_scalevar])
    online_func_body = str(tl_code)
    if online_func_body == "":
        online_func_body = "pass"
    online_func_inputs = ""
    for varname, input_var in input_vars_online.items():
        online_func_inputs += f"{input_var.varname}: T.Buffer([{', '.join(input_var.shape_idx)}], accum_dtype), \n"
    online_func_inputs_list = ", ".join(
        [input_var.varname for varname, input_var in input_vars_online.items()])
    input_vars.update(input_vars_online)
    # o_scale = o_scalevar.varname
    o_scale_varname = o_scalevar.varname
    o_scale = IndentedCode()
    o_scale.add_line(f"for i, j in T.Parallel(block_M, dimv):")
    o_scale.more_indent()
    o_scale.add_line(f"acc_o[i, j] *= {o_scalevar.varname}[i]")
    o_scale = str(o_scale)

    online_rowscales_update = ""
    for k, v in new_online_rowscales.items():
        if v.varname == k:
            continue
        online_rowscales_update += f"T.copy({v.varname}, {k})\n"

    # final_rowscales
    # final_rowscales_init = ""
    # for k,v in online_func.final_rowscales.items():
    #     final_rowscales_init += f"{k} = T.alloc_fragment([{v.shape_idx}], accum_dtype)\n"
    acco = SymbolicArray("acc_o", Var("acc_o"), shape_idx=["block_M", "dimv"])
    for k, v in online_rowscales.items():
        online_rowscales[k].clear_codegen()
    acco_new, new_final_rowscales\
        = online_func.online_fwd_epilogue(acco, online_rowscales, b, h, q_idx)
    tl_code, input_vars_final = generate_tl_from_dag(
        [acco_new] + list(new_final_rowscales.values()))
    online_func_epilogue = str(tl_code)
    input_vars.update(input_vars_final)
    final_rowscales_output = ""
    for k, v in online_func.final_rowscales.items():
        final_rowscales_output += f"g_{k}: T.Buffer([batch, heads, seq_len], accum_dtype), \n"
    final_rowscales_save = ""
    for k, v in new_final_rowscales.items():
        final_rowscales_save += f"T.copy({v.varname}, g_{k}[bz, by, bx * block_M : (bx + 1) * block_M])\n"

    online_func_init = ""
    # already in input_vars
    # for k,v in online_rowscales.items():
    #     online_func_init += f"{k} = T.alloc_fragment([{v.shape_idx}], accum_dtype)\n"
    for _, input_var in input_vars.items():
        if input_var.varname == scores_name:
            continue
        online_func_init += f"{input_var.varname} = T.alloc_fragment([{', '.join(input_var.shape_idx)}], accum_dtype)\n"

    # acco
    # acco = SymbolicArray("acco", Var("acco"), shape_idx=["block_M", "dimv"])
    # acco = acco * o_scale
    # tl_code += generate_tl(acco, varname="acco")

    # tmp_solution: mask_value
    # mask_value = "0"
    # for used in scores.use_list:
    #     if used.code.type == "ReduceMax":
    #         mask_value = "-inf"
    #         break
    # is_inf_mask = "True" if mask_value == "-inf" else "False"

    # print(tl_code)
    # print("o_scalevar:", o_scalevar.varname)
    # for k,v in new_online_rowscales.items():
    #     print(f"online_rowscale['{k}'] : {v.varname}")
    # print(input_vars)
    # print("mask_value:", mask_value)
    # print("online_func_epilogue:", online_func_epilogue)

    # bwd
    isused_doosum = False
    final_rowscales_bwd = {}
    for k, v in online_func.final_rowscales.items():
        final_rowscales_bwd[k] = SymbolScalar(
            f"{k}_shared", Var(f"{k}"), shape_idx=[
                "1", "block_N"])
    scores_2 = online_func.forward(
        SymbolScalar(
            "qkT",
            Var("qkT"),
            shape_idx=[
                "block_M",
                "block_N"]),
        final_rowscales_bwd,
        b,
        h,
        q_idx,
        SymbolScalar(
            "kv_idx",
            Var("kv_idx")))

    tl_code, input_vars_fwd = generate_tl_from_dag([scores_2])
    online_func_fwd = str(tl_code)

    dscores = online_func.backward(
        SymbolScalar(
            "dsT", Var("dsT"), shape_idx=[
                "block_M", "block_N"]), SymbolScalar(
            "qkT", Var("qkT"), shape_idx=[
                "block_M", "block_N"]), final_rowscales_bwd, SymbolScalar(
            "doosum_shared", Var("doosum_shared"), shape_idx=[
                "1", "block_N"]), b, h, q_idx, SymbolScalar(
            "kv_idx", Var("kv_idx")))

    tl_code, input_vars_bwd = generate_tl_from_dag([dscores])
    custom_bwd_body = str(tl_code)

    if "doosum_shared" in input_vars_bwd:
        isused_doosum = True

    custom_bwd_inputs = f"g_doosum: T.Buffer([batch, heads, seq_len], accum_dtype), \n" if isused_doosum else ""
    final_rowscales_shared_init = ""
    for k, v in final_rowscales_bwd.items():
        final_rowscales_shared_init += f"{v.varname} = T.alloc_shared([{', '.join(v.shape_idx)}], accum_dtype, scope='shared')\n"
    custom_bwd_inputs_init = "doosum_shared = T.alloc_shared([1, block_N], accum_dtype, scope='shared')" if isused_doosum else ""
    final_rowscales_load = ""
    for k, v in final_rowscales_bwd.items():
        final_rowscales_load += f"T.copy(g_{k}[bz, bx, k * block_N : (k + 1) * block_N], {v.varname})\n"
    custom_bwd_inputs_load = "T.copy(g_doosum[bz, bx, k * block_N : (k + 1) * block_N], doosum_shared)" if isused_doosum else ""
    final_rowscales_length = len(final_rowscales_bwd)

    return lowerOnlineFuncOutput(
        final_rowscales_output=final_rowscales_output,
        online_rowscales_initvalue=online_rowscales_initvalue,
        online_func_init=online_func_init,
        online_func_inputs=online_func_inputs,
        online_func_body=online_func_body,
        online_func_inputs_list=online_func_inputs_list,
        o_scale=o_scale,
        online_func_epilogue=online_func_epilogue,
        final_rowscales_save=final_rowscales_save,
        online_rowscales_update=online_rowscales_update,
        final_rowscales_init=None,

        isused_doosum=isused_doosum,
        final_rowscales_length=final_rowscales_length,
        final_rowscales_load=final_rowscales_load,
        online_func_fwd=online_func_fwd,
        custom_bwd_inputs_load=custom_bwd_inputs_load,
        custom_bwd_body=custom_bwd_body,
        final_rowscales_shared_init=final_rowscales_shared_init,
        custom_bwd_inputs=custom_bwd_inputs,
        custom_bwd_inputs_init=custom_bwd_inputs_init,
        o_scale_varname=o_scale_varname
    )


def lower_score_mod(score_mod, custom_fwd_inputs, lower_output: lowerOutput):
    scores = SymbolScalar(
        "scores",
        Var("scores"),
        shape_idx=[
            "block_M",
            "block_N"])
    b = SymbolScalar("b", Var("b"))
    h = SymbolScalar("h", Var("h"))
    q_idx = SymbolScalar("q_idx", Var("q_idx"))
    kv_idx = SymbolScalar("kv_idx", Var("kv_idx"))

    scores_new = score_mod(scores, custom_fwd_inputs, b, h, q_idx, kv_idx)
    tl_code, input_vars = generate_tl_from_dag([scores_new])
    score_mod_body = str(tl_code)
    score_mod_inputs = ""
    for varname, input_var in input_vars.items():
        score_mod_inputs += f"{input_var.varname}: T.Buffer([{', '.join(input_var.shape_idx)}], accum_dtype), \n"
    score_mod_inputs_list = ", ".join(
        [input_var.varname for varname, input_var in input_vars.items()])
    # print(tl_code)
    # print(input_vars)

    # backward, block_M : k, block_N : q
    qkT = SymbolScalar("qkT", Var("qkT"), shape_idx=["block_M", "block_N"])
    # # modify shape idx for
    # for k,v in custom_fwd_inputs.input_tensors.items():
    #     custom_fwd_inputs.input_tensors[k].clear_codegen()
    #     # TODO: not so ad hoc
    #     if v.shape_idx == ["block_M", "block_N"]:
    #         custom_fwd_inputs.input_tensors[k].shape_idx = ["block_N", "block_M"]
    #     elif v.shape_idx == ["block_M"]:
    #         custom_fwd_inputs.input_tensors[k].shape_idx = ["1", "block_N"]
    scores_new = score_mod(qkT, custom_fwd_inputs, b, h, q_idx, kv_idx)
    # tl_code, input_vars_fwd = generate_tl_from_dag([scores_new])
    # score_mod_inputs_bwd_list = ", ".join([varname for varname, input_var in input_vars_fwd.items()])
    scores_new.backward(
        SymbolScalar(
            "dsT",
            Var("dsT"),
            shape_idx=[
                "block_M",
                "block_N"]))
    tl_code, input_vars_fwd = generate_tl_from_dag([scores_new])
    score_mod_fwd_body = str(tl_code)
    score_mod_output_var = scores_new.varname
    score_mod_fwd_inputs = ""
    for varname, input_var in input_vars_fwd.items():
        score_mod_fwd_inputs += f"{input_var.varname}: T.Buffer([{', '.join(input_var.shape_idx)}], accum_dtype), \n"
    score_mod_inputs_bwd_list = ", ".join(
        [varname for varname, input_var in input_vars_fwd.items()])
    tl_code, input_vars = generate_tl_from_dag([qkT.grad])
    score_mod_backward = str(tl_code)
    # add forward input_vars
    for k, v in input_vars_fwd.items():
        input_vars[k] = v
    score_mod_bwd_inputs_list = ", ".join(
        [input_var.varname for varname, input_var in input_vars.items()])
    score_mod_bwd_inputs = ""
    for varname, input_var in input_vars.items():
        score_mod_bwd_inputs += f"{input_var.varname}: T.Buffer([{', '.join(input_var.shape_idx)}], accum_dtype), \n"

    score_mod_bwd_inputs_declare = ""
    score_mod_bwd_inputs_declare_shared = ""
    # TODO: dtype
    dtype = "accum_dtype"
    for varname, input_var in input_vars.items():
        if input_var.shape_idx == ["block_N", "block_M"]:
            dtype = "dtype"
            score_mod_bwd_inputs_declare_shared += f"{input_var.varname} = T.alloc_shared([{', '.join(input_var.shape_idx)}], {dtype})\n"
        else:
            score_mod_bwd_inputs_declare += f"{input_var.varname} = T.alloc_fragment([{', '.join(input_var.shape_idx)}], accum_dtype)\n"

    return lowerScoreModOutput(
        score_mod_inputs=score_mod_inputs,
        score_mod_body=score_mod_body,
        score_mod_inputs_list=score_mod_inputs_list,
        score_mod_backward=score_mod_backward,
        score_mod_bwd_inputs_list=score_mod_bwd_inputs_list,
        score_mod_bwd_inputs=score_mod_bwd_inputs,
        score_mod_inputs_bwd_list=score_mod_inputs_bwd_list,
        score_mod_fwd_inputs=score_mod_fwd_inputs,
        score_mod_fwd_body=score_mod_fwd_body,
        score_mod_output_var=score_mod_output_var,
        score_mod_bwd_inputs_declare=score_mod_bwd_inputs_declare,
        score_mod_bwd_inputs_declare_shared=score_mod_bwd_inputs_declare_shared
    )


def lower_tl(score_mod, block_mask, online_func,
             custom_fwd_inputs,
             dimqk, dimv, tl_dtype, mask_value, tuned_config=None):

    lower_output = lowerOutput()
    lower_output.tl_dtype = tl_dtype
    # TODO: mask_value: 0 or -inf
    lower_output.is_inf_mask = "True" if block_mask is not None and mask_value == "-inf" else "False"

    # tune
    if tuned_config is None:
        tune_output = TunnerOutput()
    else:
        tune_output = TunnerOutput(**tuned_config)
    scores_name = "scores"
    # TODO
    if dimv > 256:
        tune_output.block_M = "64"
        tune_output.block_N = "64"
        tune_output.stages = "1"
        tune_output.shared_fuse = "True"
    # TODO: ugly
    if tune_output.shared_fuse == "True":
        scores_name = "scores_1"

    custom_fwd_inputs_load_shared_bwd = ""
    # for k,v in custom_fwd_inputs.input_tensors.items():
    #     # modify shape
    #     shape_idx_copy = [(shape_idx_map_bwd[shape] if shape in shape_idx_map_bwd.keys() else ":") for shape in v.shape_idx]
    #     shape_idx_block = [(shape_idx_onchip_map_bwd[shape] if shape in shape_idx_onchip_map_bwd.keys() else shape) for shape in v.shape_idx]
    #     # remove "" in list
    #     shape_idx_block = [shape for shape in shape_idx_block if shape != ""] # TODO:bug[block_M] -> [1,block_M]
    #     custom_input_dtype = "accum_dtype"
    #     # load
    #     # tl copy bug when "1"
    #     if shape_idx_block == ["1"]:
    #         pass
    #         # custom_fwd_inputs_load_prolog += f"{k}[0] = g_{k}[{', '.join(shape_idx_copy)}]\n"
    #     elif not (RECURRENT_DIM in shape_idx_block):
    #         pass
    #         # custom_fwd_inputs_load_prolog += f"T.copy(g_{k}[{', '.join(shape_idx_copy)}], {k})\n"
    #     elif len(shape_idx_block) > 1 and shape_idx_block[1] != "1": # [block_N, block_M]
    #         custom_input_dtype = "dtype"
    #         custom_fwd_inputs_init += f"{k}_shared = T.alloc_shared([{', '.join(shape_idx_block)}], {custom_input_dtype})\n"
    #         custom_fwd_inputs_load_shared_bwd += f"T.copy(g_{k}[{', '.join(shape_idx_copy)}], {k}_shared)\n"
    #         custom_fwd_inputs_load_s2r += f"T.copy({k}_shared, {k})\n"
    #     else:# [block_N, 1]
    #         custom_fwd_inputs_load_shared_bwd += f"T.copy(g_{k}[{', '.join(shape_idx_copy)}], {k})\n"
    #     # TODO: dtype of custom_fwd_inputs
    #     custom_fwd_inputs_init += f"{k} = T.alloc_fragment([{', '.join(shape_idx_block)}], {custom_input_dtype})\n"
    #     custom_fwd_inputs_str += f"g_{k}: T.Buffer([{', '.join(v.shape_idx)}], {custom_input_dtype}), \n"
    #     custom_fwd_inputs.input_tensors[k].shape_idx = shape_idx_block

    custom_fwd_inputs_str = ""
    # for k,v in custom_fwd_inputs.input_tensors.items():
    #     custom_fwd_inputs_str += f"g_{k}: T.Buffer([{', '.join(v.shape_idx)}], accum_dtype), \n"
    custom_fwd_inputs_init = ""
    custom_fwd_inputs_load_prolog = ""
    custom_fwd_inputs_load_shared = ""
    custom_fwd_inputs_load_s2r = ""
    for k, v in custom_fwd_inputs.input_tensors.items():
        # modify shape
        shape_idx_copy = [(shape_idx_map[shape] if shape in shape_idx_map.keys(
        ) else ":") for shape in v.shape_idx]
        shape_idx_block = [(shape_idx_onchip_map[shape] if shape in shape_idx_onchip_map.keys(
        ) else shape) for shape in v.shape_idx]
        # remove "" in list
        # TODO:bug[block_N] -> [1,block_N]
        shape_idx_block = [shape for shape in shape_idx_block if shape != ""]
        custom_input_dtype = "accum_dtype"
        # load
        # tl copy bug when "1"
        if shape_idx_block == []:
            # shape_idx_copy = [idx_copy if idx_copy != ":" else "0" for idx_copy in shape_idx_copy]
            shape_idx_block = ["1"]
            custom_fwd_inputs_load_prolog += f"{k}[0] = g_{k}[{', '.join(shape_idx_copy)}]\n"
        elif not (RECURRENT_DIM in shape_idx_block):
            custom_fwd_inputs_load_prolog += f"T.copy(g_{k}[{', '.join(shape_idx_copy)}], {k})\n"
        elif len(shape_idx_block) > 1 and shape_idx_block[0] != "1":
            custom_input_dtype = "dtype"
            custom_fwd_inputs_init += f"{k}_shared = T.alloc_shared([{', '.join(shape_idx_block)}], {custom_input_dtype})\n"
            custom_fwd_inputs_load_shared += f"T.copy(g_{k}[{', '.join(shape_idx_copy)}], {k}_shared)\n"
            custom_fwd_inputs_load_s2r += f"T.copy({k}_shared, {k})\n"
            lower_output.swizzle_shared += f"{k}_shared: tl.layout.make_swizzled_layout({k}_shared), \n"
        else:
            custom_fwd_inputs_load_shared += f"T.copy(g_{k}[{', '.join(shape_idx_copy)}], {k})\n"
        # TODO: dtype of custom_fwd_inputs
        custom_fwd_inputs_init += f"{k} = T.alloc_fragment([{', '.join(shape_idx_block)}], {custom_input_dtype})\n"
        custom_fwd_inputs_str += f"g_{k}: T.Buffer([{', '.join(v.shape_idx)}], {custom_input_dtype}), \n"
        custom_fwd_inputs.input_tensors[k].shape_idx = shape_idx_block

    lower_score_mod_output = lower_score_mod(
        score_mod, custom_fwd_inputs, lower_output)
    lower_online_func_output = lower_online_func(
        online_func, lower_output, scores_name)
    output_idx_list = [i for i in range(3 +
                                        len(custom_fwd_inputs.input_tensors), 3 +
                                        len(custom_fwd_inputs.input_tensors) +
                                        1 +
                                        len(online_func.final_rowscales))]

    # TODO: custom_fwd_inputs grad
    bwd_output_idx_list = [i for i in range(4 +
                                            len(custom_fwd_inputs.input_tensors) +
                                            len(online_func.final_rowscales) +
                                            int(lower_online_func_output.isused_doosum), 4 +
                                            len(custom_fwd_inputs.input_tensors) +
                                            len(online_func.final_rowscales) +
                                            int(lower_online_func_output.isused_doosum) +
                                            3)]

    return TlAttnTemplate(
        custom_fwd_inputs=custom_fwd_inputs_str,
        custom_fwd_inputs_init=custom_fwd_inputs_init,
        custom_fwd_inputs_load_prolog=custom_fwd_inputs_load_prolog,
        custom_fwd_inputs_load_s2r=custom_fwd_inputs_load_s2r,
        custom_fwd_inputs_load_shared=custom_fwd_inputs_load_shared,
        custom_fwd_inputs_load_shared_bwd=custom_fwd_inputs_load_shared_bwd,
        **lower_online_func_output.__dict__,
        **lower_score_mod_output.__dict__,

        **lower_output.__dict__,
        **tune_output.__dict__,

        output_idx_list=str(output_idx_list),
        bwd_output_idx_list=str(bwd_output_idx_list)
    )()


if __name__ == "__main__":
    from mha import OnlineSoftmax
    lower_online_func(OnlineSoftmax())
    print("---------------------")
    from mha import score_mod
    custom_fwd_inputs = CustomIO({
        "softmax_scale": (1,)
    })
    lower_score_mod(score_mod, custom_fwd_inputs)

    aaa = SymbolScalar("aaa", Var("aaa"))
    b = 5.4
    a = 3
    aaa = aaa * (b / a)
    print(aaa.code)
    print(generate_tl_from_dag([aaa]))

    print("lower_tl:")
    tl_code = lower_tl(score_mod, None, OnlineSoftmax(), custom_fwd_inputs)
    with open("generated_tl_code.py", "w") as f:
        f.write(tl_code)
