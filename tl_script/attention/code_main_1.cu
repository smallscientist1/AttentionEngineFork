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
    for (int k_1 = 0; k_1 < (((int)blockIdx.x) + 1); ++k_1) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 32; ++i_4) {
        for (int vec_s = 0; vec_s < 2; ++vec_s) {
          float condval;
          if ((((((k_1 * 128) + ((i_4 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) <= ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
            condval = 0.000000e+00f;
          } else {
            condval = -CUDART_INF_F;
          }
          scores[((i_4 * 2) + vec_s)] = condval;
        }
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[(k_1 & 1)], ((k_1 & 3) >> 1));
      tl::gemm_ss<128, 128, 128, 8, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 16384) + 16384)])), (&(scores[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 4)]);
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        float2 __1;
          float2 v_ = *(float2*)(scores + (i_5 * 2));
          float2 v__1 = make_float2(1.250000e-01f, 1.250000e-01f);
          __1.x = (v_.x*v__1.x);
          __1.y = (v_.y*v__1.y);
        *(float2*)(scores + (i_5 * 2)) = __1;
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        scores_max_0[i_6] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 32; ++rv) {
          scores_max_0[i_6] = max(scores_max_0[i_6], scores[((((rv & 15) * 4) + (i_6 * 2)) + (rv >> 4))]);
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
      for (int i_11 = 0; i_11 < 32; ++i_11) {
        float2 __2;
          float2 v__2 = *(float2*)(scores + (i_11 * 2));
          float2 v__3 = make_float2(scores_max_0[(i_11 & 1)], scores_max_0[(i_11 & 1)]);
          __2.x = (v__2.x-v__3.x);
          __2.y = (v__2.y-v__3.y);
        *(float2*)(scores + (i_11 * 2)) = __2;
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 32; ++i_12) {
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
        for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
          scores_1_0_sum_0[i_13] = (scores_1_0_sum_0[i_13] + scores[((((rv_1 & 15) * 4) + (i_13 * 2)) + (rv_1 >> 4))]);
        }
        scores_1_0_sum_0[i_13] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_1_0_sum_0[i_13]);
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        r[i_14] = (r[i_14] + scores_1_0_sum_0[i_14]);
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 32; ++i_15) {
        float2 __5;
          float2 v__6 = *(float2*)(acc_o + (i_15 * 2));
          float2 v__7 = make_float2(m[(i_15 & 1)], m[(i_15 & 1)]);
          __5.x = (v__6.x*v__7.x);
          __5.y = (v__6.y*v__7.y);
        *(float2*)(acc_o + (i_15 * 2)) = __5;
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        m[i_16] = scores_max_0[i_16];
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 32; ++i_17) {
        uint1 __6;
        float2 v__8 = *(float2*)(scores + (i_17 * 2));
        ((half2*)(&(__6.x)))->x = (half_t)(v__8.x);
        ((half2*)(&(__6.x)))->y = (half_t)(v__8.y);
        *(uint1*)(acc_s_cast + (i_17 * 2)) = __6;
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 2)], ((k_1 & 3) >> 1));
      tl::gemm_rs<128, 128, 128, 8, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 16384) + 49152)])), (&(acc_o[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 6)]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 32; ++i_18) {
      float2 __7;
        float2 v__9 = *(float2*)(acc_o + (i_18 * 2));
        float2 v__10 = make_float2(r[(i_18 & 1)], r[(i_18 & 1)]);
        __7.x = (v__9.x/v__10.x);
        __7.y = (v__9.y/v__10.y);
      *(float2*)(acc_o + (i_18 * 2)) = __7;
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      r[i_19] = (__log2f(r[i_19]) * 6.931472e-01f);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      r[i_20] = (r[i_20] + m[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 32; ++i_21) {
      uint1 __8;
      float2 v__11 = *(float2*)(acc_o + (i_21 * 2));
      ((half2*)(&(__8.x)))->x = (half_t)(v__11.x);
      ((half2*)(&(__8.x)))->y = (half_t)(v__11.y);
      *(uint1*)(Output + (((((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_21 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((i_21 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __8;
    }
    if (((((int)threadIdx.x) & 3) >> 1) == 0) {
      g_lse[(((((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = r[(((int)threadIdx.x) & 1)];
    }
  }
}

