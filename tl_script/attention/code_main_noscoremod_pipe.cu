#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/ldsm.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(384) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc, float* __restrict__ g_lse) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float m[2];
  float r[2];
  float scores[64];
  float scores_max_0[2];
  float scores_1_0_sum_0[2];
  half_t acc_s_cast[64];
  __shared__ uint64_t _mbarrier[9];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 256);
    tl::mbarrier_init(_mbarrier[5], 256);
    tl::mbarrier_init(_mbarrier[6], 256);
    tl::mbarrier_init(_mbarrier[7], 256);
    tl::mbarrier_init(_mbarrier[8], 128);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) == 256) {
      tl::mbarrier_expect_tx(_mbarrier[8], 32768);
    }
    if (((int)threadIdx.x) == 256) {
      tl::tma_load(Q_desc, _mbarrier[8], (&(((half_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_load(Q_desc, _mbarrier[8], (&(((half_t*)buf_dyn_shmem)[8192])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
    }
    tl::mbarrier_arrive(_mbarrier[8]);
    for (int k = 0; k < (((int)blockIdx.x) + 1); ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 1) + 4)], (((k & 3) >> 1) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 32768);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(K_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 16384)])), 0, ((int)blockIdx.y), (k * 128), 0);
        tl::tma_load(K_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 24576)])), 64, ((int)blockIdx.y), (k * 128), 0);
      }
      tl::mbarrier_arrive(_mbarrier[(k & 1)]);
      tl::mbarrier_wait(_mbarrier[((k & 1) + 6)], (((k & 3) >> 1) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[((k & 1) + 2)], 32768);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(V_desc, _mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 49152)])), 0, ((int)blockIdx.y), (k * 128), 0);
        tl::tma_load(V_desc, _mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 16384) + 57344)])), 64, ((int)blockIdx.y), (k * 128), 0);
      }
      tl::mbarrier_arrive(_mbarrier[((k & 1) + 2)]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      *(float2*)(acc_o + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      m[i_1] = 1.000000e+00f;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      m[i_2] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      r[i_3] = 0.000000e+00f;
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[8], 0);
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      for (int vec_s = 0; vec_s < 2; ++vec_s) {
        float condval;
        if ((((((i_4 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) <= ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval = 0.000000e+00f;
        } else {
          condval = -CUDART_INF_F;
        }
        scores[((i_4 * 2) + vec_s)] = condval;
      }
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[0], 0);
    tl::gemm_ss<128, 128, 128, 8, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(scores[0])));
    tl::mbarrier_arrive(_mbarrier[4]);
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_max_0[i_5] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 32; ++rv) {
        scores_max_0[i_5] = max(scores_max_0[i_5], scores[((((rv & 15) * 4) + (i_5 * 2)) + (rv >> 4))]);
      }
      scores_max_0[i_5] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max_0[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_max_0[i_6] = max(m[i_6], scores_max_0[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      m[i_7] = (m[i_7] - scores_max_0[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      m[i_8] = exp2f((m[i_8] * 1.442695e+00f));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      r[i_9] = (r[i_9] * m[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 32; ++i_10) {
      float2 __1;
        float2 v_ = *(float2*)(scores + (i_10 * 2));
        float2 v__1 = make_float2(scores_max_0[(i_10 & 1)], scores_max_0[(i_10 & 1)]);
        __1.x = (v_.x-v__1.x);
        __1.y = (v_.y-v__1.y);
      *(float2*)(scores + (i_10 * 2)) = __1;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      float2 __2;
      float2 __3;
        float2 v__2 = *(float2*)(scores + (i_11 * 2));
        float2 v__3 = make_float2(1.442695e+00f, 1.442695e+00f);
        __3.x = (v__2.x*v__3.x);
        __3.y = (v__2.y*v__3.y);
      __2.x = exp2f(__3.x);
      __2.y = exp2f(__3.y);
      *(float2*)(scores + (i_11 * 2)) = __2;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_1_0_sum_0[i_12] = 0.000000e+00f;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
        scores_1_0_sum_0[i_12] = (scores_1_0_sum_0[i_12] + scores[((((rv_1 & 15) * 4) + (i_12 * 2)) + (rv_1 >> 4))]);
      }
      scores_1_0_sum_0[i_12] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_1_0_sum_0[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      r[i_13] = (r[i_13] + scores_1_0_sum_0[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      uint1 __4;
      float2 v__4 = *(float2*)(scores + (i_14 * 2));
      ((half2*)(&(__4.x)))->x = (half_t)(v__4.x);
      ((half2*)(&(__4.x)))->y = (half_t)(v__4.y);
      *(uint1*)(acc_s_cast + (i_14 * 2)) = __4;
    }
    for (int k_1 = 0; k_1 < ((int)blockIdx.x); ++k_1) {
      #pragma unroll
      for (int i_15 = 0; i_15 < 32; ++i_15) {
        for (int vec_s_1 = 0; vec_s_1 < 2; ++vec_s_1) {
          float condval_1;
          if (((((((k_1 * 128) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) + 128) <= ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_15 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
            condval_1 = 0.000000e+00f;
          } else {
            condval_1 = -CUDART_INF_F;
          }
          scores[((i_15 * 2) + vec_s_1)] = condval_1;
        }
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 + 1) & 1)], (((k_1 + 1) & 3) >> 1));
      tl::gemm_ss<128, 128, 128, 8, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[((((k_1 + 1) & 1) * 16384) + 16384)])), (&(scores[0])));
      tl::mbarrier_arrive(_mbarrier[(((k_1 + 1) & 1) + 4)]);
      #pragma unroll
      for (int i_16 = 0; i_16 < 32; ++i_16) {
        float2 __5;
          float2 v__5 = *(float2*)(acc_o + (i_16 * 2));
          float2 v__6 = make_float2(m[(i_16 & 1)], m[(i_16 & 1)]);
          __5.x = (v__5.x*v__6.x);
          __5.y = (v__5.y*v__6.y);
        *(float2*)(acc_o + (i_16 * 2)) = __5;
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        m[i_17] = scores_max_0[i_17];
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 2)], ((k_1 & 3) >> 1));
      tl::gemm_rs<128, 128, 128, 8, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 16384) + 49152)])), (&(acc_o[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 6)]);
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        scores_max_0[i_18] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
          scores_max_0[i_18] = max(scores_max_0[i_18], scores[((((rv_2 & 15) * 4) + (i_18 * 2)) + (rv_2 >> 4))]);
        }
        scores_max_0[i_18] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max_0[i_18]);
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        scores_max_0[i_19] = max(m[i_19], scores_max_0[i_19]);
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        m[i_20] = (m[i_20] - scores_max_0[i_20]);
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        m[i_21] = exp2f((m[i_21] * 1.442695e+00f));
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        r[i_22] = (r[i_22] * m[i_22]);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 32; ++i_23) {
        float2 __6;
          float2 v__7 = *(float2*)(scores + (i_23 * 2));
          float2 v__8 = make_float2(scores_max_0[(i_23 & 1)], scores_max_0[(i_23 & 1)]);
          __6.x = (v__7.x-v__8.x);
          __6.y = (v__7.y-v__8.y);
        *(float2*)(scores + (i_23 * 2)) = __6;
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 32; ++i_24) {
        float2 __7;
        float2 __8;
          float2 v__9 = *(float2*)(scores + (i_24 * 2));
          float2 v__10 = make_float2(1.442695e+00f, 1.442695e+00f);
          __8.x = (v__9.x*v__10.x);
          __8.y = (v__9.y*v__10.y);
        __7.x = exp2f(__8.x);
        __7.y = exp2f(__8.y);
        *(float2*)(scores + (i_24 * 2)) = __7;
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        scores_1_0_sum_0[i_25] = 0.000000e+00f;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 32; ++rv_3) {
          scores_1_0_sum_0[i_25] = (scores_1_0_sum_0[i_25] + scores[((((rv_3 & 15) * 4) + (i_25 * 2)) + (rv_3 >> 4))]);
        }
        scores_1_0_sum_0[i_25] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_1_0_sum_0[i_25]);
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 2; ++i_26) {
        r[i_26] = (r[i_26] + scores_1_0_sum_0[i_26]);
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 32; ++i_27) {
        uint1 __9;
        float2 v__11 = *(float2*)(scores + (i_27 * 2));
        ((half2*)(&(__9.x)))->x = (half_t)(v__11.x);
        ((half2*)(&(__9.x)))->y = (half_t)(v__11.y);
        *(uint1*)(acc_s_cast + (i_27 * 2)) = __9;
      }
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 32; ++i_28) {
      float2 __10;
        float2 v__12 = *(float2*)(acc_o + (i_28 * 2));
        float2 v__13 = make_float2(m[(i_28 & 1)], m[(i_28 & 1)]);
        __10.x = (v__12.x*v__13.x);
        __10.y = (v__12.y*v__13.y);
      *(float2*)(acc_o + (i_28 * 2)) = __10;
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      m[i_29] = scores_max_0[i_29];
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[((((int)blockIdx.x) & 1) + 2)], ((((int)blockIdx.x) & 3) >> 1));
    tl::gemm_rs<128, 128, 128, 8, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((((int)blockIdx.x) & 1) * 16384) + 49152)])), (&(acc_o[0])));
    tl::mbarrier_arrive(_mbarrier[((((int)blockIdx.x) & 1) + 6)]);
    #pragma unroll
    for (int i_30 = 0; i_30 < 32; ++i_30) {
      float2 __11;
        float2 v__14 = *(float2*)(acc_o + (i_30 * 2));
        float2 v__15 = make_float2(r[(i_30 & 1)], r[(i_30 & 1)]);
        __11.x = (v__14.x/v__15.x);
        __11.y = (v__14.y/v__15.y);
      *(float2*)(acc_o + (i_30 * 2)) = __11;
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      r[i_31] = (__log2f(r[i_31]) * 6.931472e-01f);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      r[i_32] = (r[i_32] + m[i_32]);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 32; ++i_33) {
      uint1 __12;
      float2 v__16 = *(float2*)(acc_o + (i_33 * 2));
      ((half2*)(&(__12.x)))->x = (half_t)(v__16.x);
      ((half2*)(&(__12.x)))->y = (half_t)(v__16.y);
      *(uint1*)(Output + (((((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_33 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((i_33 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __12;
    }
    if (((((int)threadIdx.x) & 3) >> 1) == 0) {
      g_lse[(((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = r[(((int)threadIdx.x) & 1)];
    }
  }
}

