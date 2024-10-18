// Config: 
// BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 8, 80, 1, 8192, 64, 128
// block_M, block_N, block_K, block_Dstate = 64, 64, 64, 128

// Perf: 124.13 TLFOPS
// For tl script:

// def chunk_scan_fwd(batch, seqlen, ngroups, nheads, headdim, dstate, block_M, block_N, block_K, block_Dstate):
//     dtype = "float16"
//     accum_dtype = "float"
//     nchunks = T.ceildiv(seqlen, chunk_size)
//     p = 1.44269504
//     @T.prim_func
//     def main(
//         cb: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
//         x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
//         dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
//         dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
//         C: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
//         prev_states: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype),
//         Output: T.Buffer((batch, seqlen, nheads, headdim), dtype)
//     ):
//         with T.Kernel(nheads, T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, threads=128) as (bz, bx, by):
//             acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
//             # acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
//             cb_shared = T.alloc_shared((block_M, block_K), dtype)
//             cb_local = T.alloc_fragment((block_M, block_K), dtype)
//             # cb_shared_prev = T.alloc_shared((block_M, block_K), dtype)
//             cb_local_prev = T.alloc_fragment((block_M, block_K), dtype)
//             dA_cs_k_shared = T.alloc_shared((block_M), dtype)
//             dA_cs_k_local = T.alloc_fragment((block_M), dtype)
//             dA_cs_m_shared = T.alloc_shared((block_M), dtype)
//             dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
//             dt_shared = T.alloc_shared((block_K), dtype)
//             dt_local = T.alloc_fragment((block_K), accum_dtype)
//             x_shared = T.alloc_shared((block_K, block_N), dtype)
//             scale_m_local = T.alloc_fragment((block_M), accum_dtype)
//             C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
//             prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)


//             batch_idx = by % batch
//             chunk_idx = by // batch
//             # m: chunk_size
//             # n : headdim
//             m_idx = bx // T.ceildiv(headdim, block_N)
//             n_idx = bx % T.ceildiv(headdim, block_N)

//             T.annotate_layout({
//                 # acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)
//                 cb_shared: tl.layout.make_swizzled_layout(cb_shared)
//                 # cb_shared_prev: tl.layout.make_swizzled_layout(cb_shared_prev)
//             })
            
//             T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
//             T.copy(dA_cs_m_shared, dA_cs_m_local)
//             T.clear(acc_o)
            
//             for i in T.Parallel(block_M):
//                 scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
//             T.copy(
//                 C[batch_idx, 
//                   chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
//                   bz // (nheads // ngroups),
//                   0 : block_Dstate
//                   ], 
//                 C_shared
//             )
//             T.copy(
//                 prev_states[batch_idx, 
//                   chunk_idx,
//                   bz,
//                   n_idx * block_N : (n_idx + 1) * block_N,
//                   0 : block_Dstate
//                   ], 
//                 prev_state_shared
//             )
//             T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
//             for i, j in T.Parallel(block_M, block_N):
//                 acc_o[i, j] *= scale_m_local[i]

//             loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

//             for k in T.Pipelined(loop_range, num_stages=2):
//             # for k in T.Pipelined(
//             #     loop_range, 
//             #     num_stages=2, 
//             #     order=[-1,1,-1,2,-1,3,-1,4,0], 
//             #     stage=[-1,0,-1,0,-1,0,-1,0,1], 
//             #     group=[[0],[1],[2],[3,4],[5],[6,7,8],[9],[10],[11]]
//             # ):
//                 T.copy(
//                     cb[batch_idx, 
//                        chunk_idx, 
//                        bz // (nheads // ngroups), 
//                        m_idx * block_M : (m_idx + 1) * block_M, 
//                        k * block_K : (k + 1) * block_K], 
//                     cb_shared
//                 )
//                 T.copy(cb_shared, cb_local)
//                 T.copy(
//                     dA_cumsum[batch_idx, 
//                        bz, 
//                        chunk_idx,
//                        k * block_K : (k + 1) * block_K], 
//                     dA_cs_k_shared
//                 )
//                 T.copy(dA_cs_k_shared, dA_cs_k_local)
//                 for i, j in T.Parallel(block_M, block_K):
//                     cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
//                 T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
//                 T.copy(dt_shared, dt_local)
//                 for i, j in T.Parallel(block_M, block_K):
//                     cb_local[i, j] *= dt_local[j]
//                 for i, j in T.Parallel(block_M, block_K):
//                     cb_local[i, j] = T.if_then_else(
//                         m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0
//                     )
//                 T.copy(x[batch_idx, chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, bz, n_idx * block_N : (n_idx + 1) * block_N], x_shared)
//                 # T.copy(cb_local, cb_shared_prev)
//                 T.copy(cb_local, cb_local_prev)
//                 T.gemm(cb_local_prev, x_shared, acc_o)
//             # T.copy(acc_o, acc_o_shared)
//             T.copy(acc_o, Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, bz, n_idx * block_N : (n_idx + 1) * block_N])

//     return main
#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/ldsm.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(256) main_kernel(__grid_constant__ const CUtensorMap C_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap cb_desc, __grid_constant__ const CUtensorMap dA_cumsum_desc, __grid_constant__ const CUtensorMap dt_desc, __grid_constant__ const CUtensorMap prev_states_desc, __grid_constant__ const CUtensorMap x_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ half_t dA_cs_m_shared[64];
  __shared__ half_t dA_cs_k_shared[128];
  __shared__ half_t dt_shared[128];
  float dA_cs_m_local[2];
  float acc_o[32];
  float scale_m_local[2];
  half_t cb_local[32];
  half_t dA_cs_k_local[16];
  float dt_local[16];
  half_t cb_local_prev[32];
  __shared__ uint64_t _mbarrier[18];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(dA_cumsum_desc);
    tl::prefetch_tma_descriptor(C_desc);
    tl::prefetch_tma_descriptor(prev_states_desc);
    tl::prefetch_tma_descriptor(cb_desc);
    tl::prefetch_tma_descriptor(dt_desc);
    tl::prefetch_tma_descriptor(x_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 128);
    tl::mbarrier_init(_mbarrier[5], 128);
    tl::mbarrier_init(_mbarrier[6], 128);
    tl::mbarrier_init(_mbarrier[7], 128);
    tl::mbarrier_init(_mbarrier[8], 128);
    tl::mbarrier_init(_mbarrier[9], 128);
    tl::mbarrier_init(_mbarrier[10], 128);
    tl::mbarrier_init(_mbarrier[11], 128);
    tl::mbarrier_init(_mbarrier[12], 128);
    tl::mbarrier_init(_mbarrier[13], 128);
    tl::mbarrier_init(_mbarrier[14], 128);
    tl::mbarrier_init(_mbarrier[15], 128);
    tl::mbarrier_init(_mbarrier[16], 128);
    tl::mbarrier_init(_mbarrier[17], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[16], 128);
    }
    if (((int)threadIdx.x) == 128) {
      tl::tma_load(dA_cumsum_desc, _mbarrier[16], (&(dA_cs_m_shared[0])), (((int)blockIdx.y) * 64), (((int)blockIdx.z) >> 3), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
    }
    tl::mbarrier_arrive(_mbarrier[16]);
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[17], 16384);
    }
    if (((int)threadIdx.x) == 128) {
      tl::tma_load(C_desc, _mbarrier[17], (&(((half_t*)buf_dyn_shmem)[0])), 0, 0, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 64)), (((int)blockIdx.z) & 7));
      tl::tma_load(C_desc, _mbarrier[17], (&(((half_t*)buf_dyn_shmem)[4096])), 64, 0, (((((int)blockIdx.z) >> 3) * 256) + (((int)blockIdx.y) * 64)), (((int)blockIdx.z) & 7));
    }
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[17], 16384);
    }
    if (((int)threadIdx.x) == 128) {
      tl::tma_load(prev_states_desc, _mbarrier[17], (&(((half_t*)buf_dyn_shmem)[8192])), 0, 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      tl::tma_load(prev_states_desc, _mbarrier[17], (&(((half_t*)buf_dyn_shmem)[12288])), 64, 0, ((int)blockIdx.x), (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
    }
    tl::mbarrier_arrive(_mbarrier[17]);
    for (int k = 0; k < (((int)blockIdx.y) + 1); ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 1) + 8)], ((k >> 1) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 8192);
      }
      if (((int)threadIdx.x) == 128) {
        tl::tma_load(cb_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 16384)])), (k * 64), (((int)blockIdx.y) * 64), 0, (((int)blockIdx.z) >> 3), (((int)blockIdx.z) & 7));
      }
      tl::mbarrier_arrive(_mbarrier[(k & 1)]);
      tl::mbarrier_wait(_mbarrier[((k & 1) + 10)], ((k >> 1) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[((k & 1) + 2)], 128);
      }
      if (((int)threadIdx.x) == 128) {
        tl::tma_load(dA_cumsum_desc, _mbarrier[((k & 1) + 2)], (&(dA_cs_k_shared[((k & 1) * 64)])), (k * 64), (((int)blockIdx.z) >> 3), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      tl::mbarrier_arrive(_mbarrier[((k & 1) + 2)]);
      tl::mbarrier_wait(_mbarrier[((k & 1) + 12)], ((k >> 1) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[((k & 1) + 4)], 128);
      }
      if (((int)threadIdx.x) == 128) {
        tl::tma_load(dt_desc, _mbarrier[((k & 1) + 4)], (&(dt_shared[((k & 1) * 64)])), (k * 64), (((int)blockIdx.z) >> 3), ((int)blockIdx.x), (((int)blockIdx.z) & 7));
      }
      tl::mbarrier_arrive(_mbarrier[((k & 1) + 4)]);
      tl::mbarrier_wait(_mbarrier[((k & 1) + 14)], ((k >> 1) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[((k & 1) + 6)], 8192);
      }
      if (((int)threadIdx.x) == 128) {
        tl::tma_load(x_desc, _mbarrier[((k & 1) + 6)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 24576)])), 0, ((int)blockIdx.x), (((((int)blockIdx.z) >> 3) * 256) + (k * 64)), (((int)blockIdx.z) & 7));
      }
      tl::mbarrier_arrive(_mbarrier[((k & 1) + 6)]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    tl::mbarrier_wait(_mbarrier[16], 0);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      dA_cs_m_local[i] = ((float)dA_cs_m_shared[((((((int)threadIdx.x) >> 5) * 16) + (i * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      *(float2*)(acc_o + (i_1 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      scale_m_local[i_2] = exp2f((dA_cs_m_local[i_2] * 1.442695e+00f));
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[17], 0);
    tl::gemm_ss<64, 64, 128, 4, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(acc_o[0])));
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      float2 __1;
        float2 v_ = *(float2*)(acc_o + (i_3 * 2));
        float2 v__1 = make_float2(scale_m_local[(i_3 & 1)], scale_m_local[(i_3 & 1)]);
        __1.x = (v_.x*v__1.x);
        __1.y = (v_.y*v__1.y);
      *(float2*)(acc_o + (i_3 * 2)) = __1;
    }
    for (int k_1 = 0; k_1 < (((int)blockIdx.y) + 1); ++k_1) {
      tl::mbarrier_wait(_mbarrier[(k_1 & 1)], (k_1 >> 1));
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((k_1 & 1) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(cb_local[(i_4 * 8)])));
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 8)]);
      tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 2)], (k_1 >> 1));
      #pragma unroll
      for (int i_5 = 0; i_5 < 8; ++i_5) {
        *(uint1*)(dA_cs_k_local + (i_5 * 2)) = *(uint1*)(dA_cs_k_shared + ((((k_1 & 1) * 64) + (i_5 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 10)]);
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        uint1 __2;
        float2 __3;
          float2 __4;
          uint1 v__2 = *(uint1*)(cb_local + (i_6 * 2));
          __4.x = (float)(((half2*)(&(v__2.x)))->x);
          __4.y = (float)(((half2*)(&(v__2.x)))->y);
          float2 __5;
          float2 __6;
            float2 v__3 = make_float2((dA_cs_m_local[(i_6 & 1)] * 1.442695e+00f), (dA_cs_m_local[(i_6 & 1)] * 1.442695e+00f));
            float2 __7;
              float2 __8;
              uint1 v__4 = *(uint1*)(dA_cs_k_local + ((i_6 >> 1) * 2));
              __8.x = (float)(((half2*)(&(v__4.x)))->x);
              __8.y = (float)(((half2*)(&(v__4.x)))->y);
              float2 v__5 = make_float2(1.442695e+00f, 1.442695e+00f);
              __7.x = (__8.x*v__5.x);
              __7.y = (__8.y*v__5.y);
            __6.x = (v__3.x-__7.x);
            __6.y = (v__3.y-__7.y);
          __5.x = exp2f(__6.x);
          __5.y = exp2f(__6.y);
          __3.x = (__4.x*__5.x);
          __3.y = (__4.y*__5.y);
        ((half2*)(&(__2.x)))->x = (half_t)(__3.x);
        ((half2*)(&(__2.x)))->y = (half_t)(__3.y);
        *(uint1*)(cb_local + (i_6 * 2)) = __2;
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 4)], (k_1 >> 1));
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        float2 __9;
        uint1 v__6 = *(uint1*)(dt_shared + ((((k_1 & 1) * 64) + (i_7 * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        __9.x = (float)(((half2*)(&(v__6.x)))->x);
        __9.y = (float)(((half2*)(&(v__6.x)))->y);
        *(float2*)(dt_local + (i_7 * 2)) = __9;
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 12)]);
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        uint1 __10;
        float2 __11;
          float2 __12;
          uint1 v__7 = *(uint1*)(cb_local + (i_8 * 2));
          __12.x = (float)(((half2*)(&(v__7.x)))->x);
          __12.y = (float)(((half2*)(&(v__7.x)))->y);
          float2 v__8 = *(float2*)(dt_local + ((i_8 >> 1) * 2));
          __11.x = (__12.x*v__8.x);
          __11.y = (__12.y*v__8.y);
        ((half2*)(&(__10.x)))->x = (half_t)(__11.x);
        ((half2*)(&(__10.x)))->y = (half_t)(__11.y);
        *(uint1*)(cb_local + (i_8 * 2)) = __10;
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 16; ++i_9) {
        for (int vec_s = 0; vec_s < 2; ++vec_s) {
          half_t condval;
          if ((((((k_1 * 64) + ((i_9 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) <= ((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
            condval = cb_local[((i_9 * 2) + vec_s)];
          } else {
            condval = half_t(0.000000e+00f);
          }
          cb_local[((i_9 * 2) + vec_s)] = condval;
        }
      }
      if (k_1 > 0) {
        cute::warpgroup_wait<0>();
        tl::mbarrier_arrive(_mbarrier[(((k_1 - 1) & 1) + 14)]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 16; ++i_10) {
        *(uint1*)(cb_local_prev + (i_10 * 2)) = *(uint1*)(cb_local + (i_10 * 2));
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 6)], (k_1 >> 1));
      tl::gemm_rs<64, 64, 64, 4, 1, 0, 0, -1>((&(cb_local_prev[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 4096) + 24576)])), (&(acc_o[0])));
      // cute::warpgroup_wait<0>();
      // tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 14)]);
    }
    cute::warpgroup_wait<0>();
    tl::mbarrier_arrive(_mbarrier[(((((int)blockIdx.y)) & 1) + 14)]);
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      uint1 __13;
      float2 v__9 = *(float2*)(acc_o + (i_11 * 2));
      ((half2*)(&(__13.x)))->x = (half_t)(v__9.x);
      ((half2*)(&(__13.x)))->y = (half_t)(v__9.y);
      *(uint1*)(Output + ((((((((((((int)blockIdx.z) & 7) * 41943040) + ((((int)blockIdx.z) >> 3) * 1310720)) + (((int)blockIdx.y) * 327680)) + ((((int)threadIdx.x) >> 5) * 81920)) + ((i_11 & 1) * 40960)) + (((((int)threadIdx.x) & 31) >> 2) * 5120)) + (((int)blockIdx.x) * 64)) + ((i_11 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __13;
    }
  }
}