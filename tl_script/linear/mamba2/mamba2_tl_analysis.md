# BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 8, 80, 1, 8192, 64, 128
block_M, block_N, block_K, block_Dstate = 64, 64, 64, 128

chunk_scan_fwd: triton 2 ms, tl 1.6 ms tl2 1.08ms
chunk_state_fwd: triton 0.77ms tl 0.66 ms tl2 0.40 ms

# tl nofuse, chunk_o
BT 64 : 64, 64, 64, 2, 128 
1.26ms  --> 1.04 ms
64, 64, 64, 1, 128 
<!--(DV 128) 5.50ms no cumsum: 5.45 ms  torch.empty h: 2.58ms   chunk_fwd h: 4.30ms (chunk_h: 1.7 ms  chunk_o: 2.6 ms) -->
1.27ms
<!--(DV 128) mamba2 chunk_scan_fwd: 4.4 ms (not 2 ms, CB float32, dt float32, dA_cumsum float32, states float32), chunk_state_fwd: 2 ms -->

BT 256: 256, 64, 64, 2, 128*2
16.5 ms
256, 128, 64, 1, 128 (stage for on NK)
35.4 ms
 256, 64, 64, 2, 128 
 41.3 ms
256, 128, 64, 1, 128*2
15.1 ms

# tl nofuse, chunk_h
BT 64 : 64, 64, 64, 2, 128
0.85 ms --> 0.79 ms
BT 256: 256, 64, 64, 2, 128
0.63 ms --> 0.49 ms

# tl fuse v_mod chunk_o
BT 64 : 64, 64, 64, 2, 128 
1.53 ms --> 1.11 ms --vmod2kmod--> 1.06 ms

# tl fuse v_mod chunk_h
BT 64 : 64, 64, 64, 2, 128
1.78 ms --> 1.35 ms --vmod2kmod--> 0.81 ms


# e2e mamba op
tl 2.07 ms, tl no fuse 2.89 ms , mamba2 3.89 ms 
tl-- tile_new--> 1.94 ms 


