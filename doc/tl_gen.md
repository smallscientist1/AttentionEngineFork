# tl code gen
未采用 op_list ，按顺序生成；
而是采用DAG，递归的生成

# Attn performance(need to be optimized?)

head 128, causal
145 TFlops(tl) vs 169 TFlops(fa2)
