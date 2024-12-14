# linear attention

standard attention can be expressed as 
```py
# query: [B, H, q_len, DK]
# key: [B, H, kv_len, DK]
# value: [B, H, kv_len, DV]
scores = query_mod(query) @ key_mod(key)
scores = block_mask(scores)
scores = score_mod(scores)
p = online_func(scores)
o = p @ value_mod(value)
```

When online_func is Identity, and score_mod&block_mask is a lower triangular matrix with certain constraint, the attention can be expressed as linear attention and has recurrent form.

Linear Attention has a more performant implementation for forward&backward.
User can define their own linear attention as follows:
```py
# q_1, ..., q_T = query[:,:,:T,:] # query [batch,head,T,D]
# k_1, ..., k_T = key[:,:,:T,:] # key [batch,head,T,D]
# v_1, ..., v_T = value[:,:,:T,:] # value [batch, head, T, DV]
# decay_1, ..., decay_T = decay[:,:,:T] # decay [batch, head, T]
h_i = h_{i-1} * exp(decay_mod(decay)_i) + K_mod(K_i) @ V_mod(V_i)
o_i = Q_mod(q_i) @ h_i
```

## mamba 2
```py
# decay [batch,heads,T]
# A [heads]
def decay_mod(decay):
    return decay*A

def k_mod(k):
    return k*dt
```

## Simple GLA
```py
def decay_mod (decay):
    return decay
def Q_mod(q):
    return q / sqrt(DK)
```

## retention
```py
def decay_mod(decay):
    return log(decay)
def Q_mod(q):
    return q / sqrt(DK)

```

## GLA
```py
# decay: [B, H, T, DK]
def decay_mod(decay):
    return decay
```
