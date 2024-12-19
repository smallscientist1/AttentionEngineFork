BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 16, 8, 8, 2048, 128, 128 

    block_M, block_N, block_K, block_Dstate = 64, 128, 64, 128

chunk_scan: 0.11 ms,


    BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 16, 8, 8, 2048, 128, 128 
    block_M, block_N, block_K = 128, 128, 64

chunk_state: 0.06 ms

    BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 16, 8, 8, 2048, 128, 128 
    block_M, block_N, block_K, block_Dstate = 64, 64, 128, 128

bmm_chunk: 0.12 ms



simple gla:
tl 0.27 ms, gla 0.3 ms 
