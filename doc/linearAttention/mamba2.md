# linear attention
input a sequence V [T, P],
output a sequence Y [T, P]

with strucutured matrix M [T,T] , Y = M V

typical M [T,T] forms: (L * (Q @ K^T)), Q[T,N] K[T,N]

# dual form of algorthm
- chunk-wise
```

```

- recurrent
```py
i = 1,2,...,seqlen

# state_i [batch,head,dstate,d] decay_i [batch,head, (dstate)]
# B_i [batch,head,dstate,1] x_i[batch,head,1,d]
state_i = state_{i-1} * exp(decay_i) + B_i @ x_i 
# y_i[batch,head,1,d] C_i [batch,head,1,dstate]  state_i [batch,head,dstate,d]
y_i = C_i @ state_i
```


# efficient kernel
- train: chunk-wise to leverage tensorcore
- inference: recurrent for linear complexity


# examples
## Mamba SSD block
- chunk
```py
input: C(Q), B(K), x(V), A(=> L)
C [batch,seqlen, nheads, dstates]
B [batch,seqlen, nheads, dstates]
x [batch,seqlen, nheads, d]
A [batch,seqlen, nheads]

output: Y
Y [batch,seqlen, nheads, d]

# some elementwise
A = A * dt
x = x * dt

# chunks 
x, A, B, C = rearrange("b (c l) ... -> b c l ...", x, A, B, C)

# generate L for diagonal
A = rearrange(A, "b c l h -> b h c l") # [batch, nheads, nchunk, chunk_size]
A_cumsum = torch.cumsum(A, dim=-1)
L = torch.exp(segsum(A, device=device)) # [batch, nheads, nchunk, chunk_size, chunk_size] 下三角矩阵

Y_diag = (L * (C @ B ) ) @ x # [batch, nchunk, chunk_size, head, d]

# generate intra-chunk scale for off-diagonal
decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum) # [batch, nheads, nchunk, chunk_size]
states =  (B*decay_states) @ x # [batch, nchunk, nheads, d, dstate]

# interchunk recurrence
states = cat(initial_states, states, dim=1) # [batch, nchunk+1, nheads, d, dstate]
decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device)) # [batch, nheads, nchunk+1, nchunk+1]
new_states = decay_chunk @ states # [batch, nchunk+1, nheads, d, dstate]
states, final_state = new_states[:, :-1], new_states[:, -1] # [batch, nchunk, nheads, d, dstate], [batch, nheads, d, dstate]

# output Y_off   off_diagonal
state_decay_out = torch.exp(A_cumsum) # [batch, nheads, nchunk, chunk_size]
Y_off = (C @ states) * state_decay_out # [batch, nchunk, chunk_size, nheads, d]

Y = Y_diag + Y_off
```

## retention

## GLA

- chunk
```py
input Q, K, V, g
Q [batch, head, seqlen, dstate]
K [batch, head, seqlen, dstate]
V [batch, head, seqlen, d]
G [batch, head, seqlen, dstate]
scale = 1 / sqrt(dstate)

output
Y 

# chunk

# elementwise

# 
state [batch, head, seqlen, nchunk]

state += 
...

# fused chunk

# inter chunk, grid(NV,NK,B*H)
for nchunk:
    Y_out = Q @ state # [Batch, head, _, chunk_size, d]
    state = state * tl.exp(d_b)[:, None] + k@v # batch,_, head, dstate, d
    Y_out.advance()

# intra chunk, grid (NK,NT,B*H)
v: batch,head,nchunk,chunk_size,d
# elementwise

# intrachunk
scores = sum(q * k * scale) # [B, H, nchunk, chunk_size, chunk_size]]
Y_diag = scores @ v # B,H,nchunk, chunk_size, d

Y = Y_out + Y_diag
final_state = state



```
