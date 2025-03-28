# AttentionEngine API


## Custom Attention Level API
This level API is designed for users to define their own attention mechanism. Users needs to define the following components:
- `qkv_meta`: meta information for Q, K, V, such as shape, dtype
- `score_mod`: elementwise modification on attention scores
- `mask_mod`: mask modification on attention scores
- `online_func`: online function for attention scores
- `custom_fwd_inputs`: custom inputs for attention mechanism
```py
mod = AttentionEngine(
    qkv_meta: Tuple[MetaTensor, MetaTensor, MetaTensor],
    score_mod: Callable[[Tensor, CustomIO, int, int, int, int], Tensor],
    custom_fwd_inputs: CustomIO,
    online_func: OnlineFunc,
    mask_mod: Callable[[Tensor, CustomIO, int, int, int, int], Tensor]
)
```

For compiled Attention module, users can use pytorch-compatible API to interact with the module.
```py
output = mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    custom_inputs: Optional[List[torch.Tensor]]
)
output.backward(do)
```

### OnlineFunc

OnlineFunc is a class that defines the online function for attention scores, such as online softmax and retention.

Details can be found in the [Getting-started Example](./getting_started_example.md).

### Customized Linear Attention API

AttentionEngine also support customized linear attention mechanism. Users can define their own linear attention mechanism by defining the following components:
- `qkv_meta`: meta information for Q, K, V, such as shape, dtype
- `q_mod`: elementwise modification on Q
- `k_mod`: elementwise modification on K
- `v_mod`: elementwise modification on V
- `decay_mod`: elementwise modification on decay
- `custom_io`: custom inputs for attention mechanism
```py
mod = LinearAttentionEngine(
    qkv_meta: Tuple[MetaTensor, MetaTensor, MetaTensor],
    q_mod: Callable[[Tensor, CustomIO], Tensor],
    k_mod: Callable[[Tensor, CustomIO], Tensor],
    v_mod: Callable[[Tensor, CustomIO], Tensor],
    decay_mod: Callable[[Tensor, CustomIO], Tensor],
    custom_io: CustomIO
)
output = mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    custom_io: Optional[List[torch.Tensor]]
)
output.backward(do)
```



# Upcoming Features 

## Attention Library Level API
This level API is designed for users to use the existing attention mechanism in the library. 
```py
mod = AttentionLibrary(
    attn_type: str="SoftmaxAttention",
    mask_type: str="Causal",
    use_types: str="Train",
)
```

## Custom Attention Level API
- Support for varlen, block-sparse mask and block-sparse indices
```py
mod = AttentionEngine(
    ...
)
mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_len_q: torch.Tensor=None,
    cu_len_kv: torch.Tensor=None,
    block_sparse_mask: torch.Tensor=None,
    block_sparse_indices: torch.Tensor=None,
    block_num: torch.Tensor=None,
)
```
- Support for OnlineFunc with splitk decoding


