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
  float scores[32];
  float scores_max_0[2];
  float scores_1_0_sum_0[2];
  half_t acc_s_cast[32];
  __shared__ uint64_t _mbarrier[13];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 128);
    tl::mbarrier_init(_mbarrier[5], 128);
    tl::mbarrier_init(_mbarrier[6], 256);
    tl::mbarrier_init(_mbarrier[7], 256);
    tl::mbarrier_init(_mbarrier[8], 256);
    tl::mbarrier_init(_mbarrier[9], 256);
    tl::mbarrier_init(_mbarrier[10], 256);
    tl::mbarrier_init(_mbarrier[11], 256);
    tl::mbarrier_init(_mbarrier[12], 128);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) == 256) {
      tl::mbarrier_expect_tx(_mbarrier[12], 49152);
    }
    if (((int)threadIdx.x) == 256) {
      tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[8192])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_load(Q_desc, _mbarrier[12], (&(((half_t*)buf_dyn_shmem)[16384])), 128, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
    }
    tl::mbarrier_arrive(_mbarrier[12]);
    for (int k = 0; k < ((((int)blockIdx.x) * 2) + 2); ++k) {
      tl::mbarrier_wait(_mbarrier[((k % 3) + 6)], (((k % 6) / 3) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[(k % 3)], 24576);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(K_desc, _mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 12288) + 49152)])), 0, ((int)blockIdx.y), (k * 64), 0);
        tl::tma_load(K_desc, _mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 12288) + 53248)])), 64, ((int)blockIdx.y), (k * 64), 0);
        tl::tma_load(K_desc, _mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 12288) + 57344)])), 128, ((int)blockIdx.y), (k * 64), 0);
      }
      tl::mbarrier_arrive(_mbarrier[(k % 3)]);
      tl::mbarrier_wait(_mbarrier[((k % 3) + 9)], (((k % 6) / 3) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[((k % 3) + 3)], 16384);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(V_desc, _mbarrier[((k % 3) + 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 8192) + 24576)])), 0, ((int)blockIdx.y), (k * 64), 0);
        tl::tma_load(V_desc, _mbarrier[((k % 3) + 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 8192) + 28672)])), 64, ((int)blockIdx.y), (k * 64), 0);
      }
      tl::mbarrier_arrive(_mbarrier[((k % 3) + 3)]);
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
    tl::mbarrier_wait(_mbarrier[12], 0);
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
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
    tl::gemm_ss<128, 64, 192, 8, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[49152])), (&(scores[0])));
    tl::mbarrier_arrive(_mbarrier[6]);
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      float2 __1;
        float2 v_ = *(float2*)(scores + (i_5 * 2));
        float2 v__1 = make_float2(7.216878e-02f, 7.216878e-02f);
        __1.x = (v_.x*v__1.x);
        __1.y = (v_.y*v__1.y);
      *(float2*)(scores + (i_5 * 2)) = __1;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_max_0[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max_0[i_6] = max(scores_max_0[i_6], scores[((((rv & 7) * 4) + (i_6 * 2)) + (rv >> 3))]);
      }
      scores_max_0[i_6] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max_0[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_max_0[i_7] = max(m[i_7], scores_max_0[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      m[i_8] = (m[i_8] - scores_max_0[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      m[i_9] = exp2f((m[i_9] * 1.442695e+00f));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      r[i_10] = (r[i_10] * m[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      float2 __2;
        float2 v__2 = *(float2*)(scores + (i_11 * 2));
        float2 v__3 = make_float2(scores_max_0[(i_11 & 1)], scores_max_0[(i_11 & 1)]);
        __2.x = (v__2.x-v__3.x);
        __2.y = (v__2.y-v__3.y);
      *(float2*)(scores + (i_11 * 2)) = __2;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      float2 __3;
      float2 __4;
        float2 v__4 = *(float2*)(scores + (i_12 * 2));
        float2 v__5 = make_float2(1.442695e+00f, 1.442695e+00f);
        __4.x = (v__4.x*v__5.x);
        __4.y = (v__4.y*v__5.y);
      __3.x = exp2f(__4.x);
      __3.y = exp2f(__4.y);
      *(float2*)(scores + (i_12 * 2)) = __3;
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      scores_1_0_sum_0[i_13] = 0.000000e+00f;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_1_0_sum_0[i_13] = (scores_1_0_sum_0[i_13] + scores[((((rv_1 & 7) * 4) + (i_13 * 2)) + (rv_1 >> 3))]);
      }
      scores_1_0_sum_0[i_13] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_1_0_sum_0[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      r[i_14] = (r[i_14] + scores_1_0_sum_0[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      uint1 __5;
      float2 v__6 = *(float2*)(scores + (i_15 * 2));
      ((half2*)(&(__5.x)))->x = (half_t)(v__6.x);
      ((half2*)(&(__5.x)))->y = (half_t)(v__6.y);
      *(uint1*)(acc_s_cast + (i_15 * 2)) = __5;
    }
    for (int k_1 = 0; k_1 < ((((int)blockIdx.x) * 2) + 1); ++k_1) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 16; ++i_16) {
        for (int vec_s_1 = 0; vec_s_1 < 2; ++vec_s_1) {
          float condval_1;
          if (((((((k_1 * 64) + ((i_16 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) + 64) <= ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_16 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
            condval_1 = 0.000000e+00f;
          } else {
            condval_1 = -CUDART_INF_F;
          }
          scores[((i_16 * 2) + vec_s_1)] = condval_1;
        }
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 + 1) % 3)], (((k_1 + 1) % 6) / 3));
      tl::gemm_ss<128, 64, 192, 8, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[((((k_1 + 1) % 3) * 12288) + 49152)])), (&(scores[0])));
      tl::mbarrier_arrive(_mbarrier[(((k_1 + 1) % 3) + 6)]);
      #pragma unroll
      for (int i_17 = 0; i_17 < 32; ++i_17) {
        float2 __6;
          float2 v__7 = *(float2*)(acc_o + (i_17 * 2));
          float2 v__8 = make_float2(m[(i_17 & 1)], m[(i_17 & 1)]);
          __6.x = (v__7.x*v__8.x);
          __6.y = (v__7.y*v__8.y);
        *(float2*)(acc_o + (i_17 * 2)) = __6;
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        m[i_18] = scores_max_0[i_18];
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 % 3) + 3)], ((k_1 % 6) / 3));
      tl::gemm_rs<128, 128, 64, 8, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 % 3) * 8192) + 24576)])), (&(acc_o[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 % 3) + 9)]);
      #pragma unroll
      for (int i_19 = 0; i_19 < 16; ++i_19) {
        float2 __7;
          float2 v__9 = *(float2*)(scores + (i_19 * 2));
          float2 v__10 = make_float2(7.216878e-02f, 7.216878e-02f);
          __7.x = (v__9.x*v__10.x);
          __7.y = (v__9.y*v__10.y);
        *(float2*)(scores + (i_19 * 2)) = __7;
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        scores_max_0[i_20] = -CUDART_INF_F;
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
          scores_max_0[i_20] = max(scores_max_0[i_20], scores[((((rv_2 & 7) * 4) + (i_20 * 2)) + (rv_2 >> 3))]);
        }
        scores_max_0[i_20] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max_0[i_20]);
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        scores_max_0[i_21] = max(m[i_21], scores_max_0[i_21]);
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        m[i_22] = (m[i_22] - scores_max_0[i_22]);
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        m[i_23] = exp2f((m[i_23] * 1.442695e+00f));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        r[i_24] = (r[i_24] * m[i_24]);
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 16; ++i_25) {
        float2 __8;
          float2 v__11 = *(float2*)(scores + (i_25 * 2));
          float2 v__12 = make_float2(scores_max_0[(i_25 & 1)], scores_max_0[(i_25 & 1)]);
          __8.x = (v__11.x-v__12.x);
          __8.y = (v__11.y-v__12.y);
        *(float2*)(scores + (i_25 * 2)) = __8;
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 16; ++i_26) {
        float2 __9;
        float2 __10;
          float2 v__13 = *(float2*)(scores + (i_26 * 2));
          float2 v__14 = make_float2(1.442695e+00f, 1.442695e+00f);
          __10.x = (v__13.x*v__14.x);
          __10.y = (v__13.y*v__14.y);
        __9.x = exp2f(__10.x);
        __9.y = exp2f(__10.y);
        *(float2*)(scores + (i_26 * 2)) = __9;
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 2; ++i_27) {
        scores_1_0_sum_0[i_27] = 0.000000e+00f;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
          scores_1_0_sum_0[i_27] = (scores_1_0_sum_0[i_27] + scores[((((rv_3 & 7) * 4) + (i_27 * 2)) + (rv_3 >> 3))]);
        }
        scores_1_0_sum_0[i_27] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_1_0_sum_0[i_27]);
      }
      #pragma unroll
      for (int i_28 = 0; i_28 < 2; ++i_28) {
        r[i_28] = (r[i_28] + scores_1_0_sum_0[i_28]);
      }
      #pragma unroll
      for (int i_29 = 0; i_29 < 16; ++i_29) {
        uint1 __11;
        float2 v__15 = *(float2*)(scores + (i_29 * 2));
        ((half2*)(&(__11.x)))->x = (half_t)(v__15.x);
        ((half2*)(&(__11.x)))->y = (half_t)(v__15.y);
        *(uint1*)(acc_s_cast + (i_29 * 2)) = __11;
      }
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 32; ++i_30) {
      float2 __12;
        float2 v__16 = *(float2*)(acc_o + (i_30 * 2));
        float2 v__17 = make_float2(m[(i_30 & 1)], m[(i_30 & 1)]);
        __12.x = (v__16.x*v__17.x);
        __12.y = (v__16.y*v__17.y);
      *(float2*)(acc_o + (i_30 * 2)) = __12;
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      m[i_31] = scores_max_0[i_31];
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[((((((int)blockIdx.x) * 2) + 1) % 3) + 3)], ((((((int)blockIdx.x) % 3) * 2) + 1) / 3));
    tl::gemm_rs<128, 128, 64, 8, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((((((int)blockIdx.x) * 2) + 1) % 3) * 8192) + 24576)])), (&(acc_o[0])));
    tl::mbarrier_arrive(_mbarrier[((((((int)blockIdx.x) * 2) + 1) % 3) + 9)]);
    #pragma unroll
    for (int i_32 = 0; i_32 < 32; ++i_32) {
      float2 __13;
        float2 v__18 = *(float2*)(acc_o + (i_32 * 2));
        float2 v__19 = make_float2(r[(i_32 & 1)], r[(i_32 & 1)]);
        __13.x = (v__18.x/v__19.x);
        __13.y = (v__18.y/v__19.y);
      *(float2*)(acc_o + (i_32 * 2)) = __13;
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 2; ++i_33) {
      r[i_33] = (__log2f(r[i_33]) * 6.931472e-01f);
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 2; ++i_34) {
      r[i_34] = (r[i_34] + m[i_34]);
    }
    #pragma unroll
    for (int i_35 = 0; i_35 < 32; ++i_35) {
      uint1 __14;
      float2 v__20 = *(float2*)(acc_o + (i_35 * 2));
      ((half2*)(&(__14.x)))->x = (half_t)(v__20.x);
      ((half2*)(&(__14.x)))->y = (half_t)(v__20.y);
      *(uint1*)(Output + (((((((((int)blockIdx.x) * 262144) + ((((int)threadIdx.x) >> 5) * 32768)) + ((i_35 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.y) * 128)) + ((i_35 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __14;
    }
    if (((((int)threadIdx.x) & 3) >> 1) == 0) {
      g_lse[(((((((int)blockIdx.y) * 2048) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = r[(((int)threadIdx.x) & 1)];
    }
  }
}

