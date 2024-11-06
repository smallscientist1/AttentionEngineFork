from core import OnlineFunc, SymbolScalar, SymbolicArray
from graph import Var
from utils import IndentedCode

def to_tl_op(type:str, *args:SymbolScalar):
    code = IndentedCode()
    if type == "ReduceSum":
        code.add_line(
            f"T.reduce_sum({args[1].varname}, {args[0].varname},dim=1, clear=True)"
        )
    elif type == "ReduceMax":
        code.add_line(
            f"T.reduce_max({args[1].varname}, {args[0].varname},dim=1, clear=True)"
        )
    elif type == "Sub" or type == "Add" or type == "Mul" or type == "Div" or type == "Neg" or type == "Exp" or type == "Log" or type == "Abs" or type == "Max":
        # args idx
        # note: assume input shape is validate: ["1",...] or [arg0[0], ...]
        idx_strs = []
        for _, arg in enumerate(args):
            input_idx = arg.shape_idx
            idx_str = [f"i{i}" if idx!="1" else f"0" for i, idx in enumerate(input_idx)]
            idx_str = ",".join(idx_str)
            idx_strs.append(idx_str)
        # [block_M,block_N]
        loop_str = ",".join(args[0].shape_idx)
        idx_str = idx_strs[0]

        # for loop
        code.add_line(
            f"for {idx_str} in T.Parallel({loop_str}):"
        )
        code.more_indent()

        if type == "Sub":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}[{idx_strs[1]}] - {args[2].varname}[{idx_strs[2]}]"
            )
        elif type == "Add":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}[{idx_strs[1]}] + {args[2].varname}[{idx_strs[2]}]"
            )
        elif type == "Max":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.max({args[1].varname}[{idx_strs[1]}], {args[2].varname}[{idx_strs[2]}])"
            )
        elif type == "Exp":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.exp2({args[1].varname}[{idx_strs[1]}]*1.442695)"
            )
        elif type == "Mul":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}[{idx_strs[1]}] * {args[2].varname}[{idx_strs[2]}]"
            )
        else: # TODO
            raise NotImplementedError(str(type))
        
        code.less_indent()
    else:
        raise NotImplementedError
    return code

def generate_tl(x:SymbolScalar):
    # print(x.varname)
    # print(type(x))
    tl_code = IndentedCode()
    if isinstance(x.code, Var):
        tl_code.add_line(f"{x.varname} = {x.code.name}")
        return tl_code
    # for i, input_item in enumerate(x.code.inputs):
    #     tl_code += generate_tl(SymbolScalar(f"{x.varname}_i{i}", input_item))
    for i, input_item in enumerate(x.prev):
        tl_code += generate_tl(input_item)

    # tl_code += to_tl_op(x.code.type, x, *[SymbolScalar(f"{x.varname}_i{i}", input_item) for i, input_item in enumerate(x.code.inputs)])
    tl_code += to_tl_op(x.code.type, x, *x.prev)
    return tl_code

def lower_online_func(online_func: OnlineFunc):
    online_fwd = online_func.online_fwd
    scores = SymbolicArray("scores", Var("scores"), shape_idx=["block_M", "block_N"])
    online_rowscales = online_func.online_rowscales
    b = SymbolScalar("b", Var("b"))
    h = SymbolScalar("h", Var("h"))
    q_idx = SymbolScalar("q_idx", Var("q_idx"))
    
    # modify online_rowscales
    # TODO: generate init
    for k,v in online_rowscales.items():
        online_rowscales[k] = SymbolScalar(k, Var(k))
    scores_new, new_online_rowscales, o_scale = online_fwd(scores, online_rowscales, b, h, q_idx)
    # print(scores_new.code)
    # for k,v in new_online_rowscales.items():
    #     print(f"{k} = {v.code}")
    # print(o_scale.code)
    tl_code = generate_tl(new_online_rowscales["r"])
    tl_code += generate_tl(scores_new)
    print(tl_code)

if __name__ == "__main__":
    from mha import OnlineSoftmax
    lower_online_func(OnlineSoftmax())

    