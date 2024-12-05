#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/ldsm.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(384) main_kernel(__grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc, float* __restrict__ g_softmax_bias) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float softmax_bias[1];
  float acc_o[64];
  float o_scale[2];
  float scores[64];
  half_t acc_s_cast[64];
  __shared__ uint64_t _mbarrier[12];
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
    tl::mbarrier_init(_mbarrier[9], 256);
    tl::mbarrier_init(_mbarrier[10], 128);
    tl::mbarrier_init(_mbarrier[11], 128);
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
    for (int k = 0; k < ((int)blockIdx.x); ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 1) + 4)], (((k & 3) >> 1) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 32768);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(K_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[16384])), 0, ((int)blockIdx.y), (k * 128), 0);
        tl::tma_load(K_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[24576])), 64, ((int)blockIdx.y), (k * 128), 0);
      }
      tl::mbarrier_arrive(_mbarrier[(k & 1)]);
      tl::mbarrier_wait(_mbarrier[((k & 1) + 6)], (((k & 3) >> 1) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[((k & 1) + 2)], 32768);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(V_desc, _mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[32768])), 0, ((int)blockIdx.y), (k * 128), 0);
        tl::tma_load(V_desc, _mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[40960])), 64, ((int)blockIdx.y), (k * 128), 0);
      }
      tl::mbarrier_arrive(_mbarrier[((k & 1) + 2)]);
    }
    tl::mbarrier_wait(_mbarrier[9], 0);
    if (((int)threadIdx.x) == 256) {
      tl::mbarrier_expect_tx(_mbarrier[10], 32768);
    }
    if (((int)threadIdx.x) == 256) {
      tl::tma_load(K_desc, _mbarrier[10], (&(((half_t*)buf_dyn_shmem)[16384])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_load(K_desc, _mbarrier[10], (&(((half_t*)buf_dyn_shmem)[24576])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
    }
    tl::mbarrier_arrive(_mbarrier[10]);
    if (((int)threadIdx.x) == 256) {
      tl::mbarrier_expect_tx(_mbarrier[11], 32768);
    }
    if (((int)threadIdx.x) == 256) {
      tl::tma_load(V_desc, _mbarrier[11], (&(((half_t*)buf_dyn_shmem)[32768])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_load(V_desc, _mbarrier[11], (&(((half_t*)buf_dyn_shmem)[40960])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
    }
    tl::mbarrier_arrive(_mbarrier[11]);
  } else {
    tl::warpgroup_reg_alloc<240>();
    softmax_bias[0] = g_softmax_bias[0];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      *(float2*)(acc_o + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      o_scale[i_1] = 1.000000e+00f;
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[8], 0);
    for (int k_1 = 0; k_1 < ((int)blockIdx.x); ++k_1) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 32; ++i_2) {
        for (int vec_s = 0; vec_s < 2; ++vec_s) {
          float condval;
          if ((((((k_1 * 128) + ((i_2 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) <= ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
            condval = 0.000000e+00f;
          } else {
            condval = -CUDART_INF_F;
          }
          scores[((i_2 * 2) + vec_s)] = condval;
        }
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[(k_1 & 1)], ((k_1 & 3) >> 1));
      tl::gemm_ss<128, 128, 128, 8, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(scores[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 4)]);
      #pragma unroll
      for (int i_3 = 0; i_3 < 32; ++i_3) {
        float2 __1;
          float2 v_ = *(float2*)(scores + (i_3 * 2));
          float2 v__1 = make_float2(softmax_bias[0], softmax_bias[0]);
          __1.x = (v_.x+v__1.x);
          __1.y = (v_.y+v__1.y);
        *(float2*)(scores + (i_3 * 2)) = __1;
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 32; ++i_4) {
        float2 __2;
          float2 v__2 = *(float2*)(scores + (i_4 * 2));
          float2 v__3 = make_float2(5.000000e-01f, 5.000000e-01f);
          __2.x = (v__2.x*v__3.x);
          __2.y = (v__2.y*v__3.y);
        *(float2*)(scores + (i_4 * 2)) = __2;
      }
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_5 = 0; i_5 < 64; ++i_5) {
        fasttanh((&(scores[i_5])), (&(scores[i_5])));
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 32; ++i_6) {
        float2 __3;
          float2 v__4 = *(float2*)(scores + (i_6 * 2));
          float2 v__5 = make_float2(1.000000e+00f, 1.000000e+00f);
          __3.x = (v__4.x+v__5.x);
          __3.y = (v__4.y+v__5.y);
        *(float2*)(scores + (i_6 * 2)) = __3;
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 32; ++i_7) {
        float2 __4;
          float2 v__6 = *(float2*)(scores + (i_7 * 2));
          float2 v__7 = make_float2(5.000000e-01f, 5.000000e-01f);
          __4.x = (v__6.x*v__7.x);
          __4.y = (v__6.y*v__7.y);
        *(float2*)(scores + (i_7 * 2)) = __4;
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 32; ++i_8) {
        float2 __5;
          float2 v__8 = *(float2*)(acc_o + (i_8 * 2));
          float2 v__9 = make_float2(o_scale[(i_8 & 1)], o_scale[(i_8 & 1)]);
          __5.x = (v__8.x*v__9.x);
          __5.y = (v__8.y*v__9.y);
        *(float2*)(acc_o + (i_8 * 2)) = __5;
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 32; ++i_9) {
        uint1 __6;
        float2 v__10 = *(float2*)(scores + (i_9 * 2));
        ((half2*)(&(__6.x)))->x = (half_t)(v__10.x);
        ((half2*)(&(__6.x)))->y = (half_t)(v__10.y);
        *(uint1*)(acc_s_cast + (i_9 * 2)) = __6;
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 2)], ((k_1 & 3) >> 1));
      tl::gemm_rs<128, 128, 128, 8, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[32768])), (&(acc_o[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 6)]);
    }
    tl::mbarrier_arrive(_mbarrier[9]);
    #pragma unroll
    for (int i_10 = 0; i_10 < 32; ++i_10) {
      for (int vec_s_1 = 0; vec_s_1 < 2; ++vec_s_1) {
        float condval_1;
        if ((((((i_10 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + vec_s_1) <= ((((((int)threadIdx.x) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
          condval_1 = 0.000000e+00f;
        } else {
          condval_1 = -CUDART_INF_F;
        }
        scores[((i_10 * 2) + vec_s_1)] = condval_1;
      }
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[10], 0);
    tl::gemm_ss<128, 128, 128, 8, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(scores[0])));
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      float2 __7;
        float2 v__11 = *(float2*)(scores + (i_11 * 2));
        float2 v__12 = make_float2(softmax_bias[0], softmax_bias[0]);
        __7.x = (v__11.x+v__12.x);
        __7.y = (v__11.y+v__12.y);
      *(float2*)(scores + (i_11 * 2)) = __7;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      float2 __8;
        float2 v__13 = *(float2*)(scores + (i_12 * 2));
        float2 v__14 = make_float2(5.000000e-01f, 5.000000e-01f);
        __8.x = (v__13.x*v__14.x);
        __8.y = (v__13.y*v__14.y);
      *(float2*)(scores + (i_12 * 2)) = __8;
    }
    tl::fence_proxy_async();
    #pragma unroll
    for (int i_13 = 0; i_13 < 64; ++i_13) {
      fasttanh((&(scores[i_13])), (&(scores[i_13])));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      float2 __9;
        float2 v__15 = *(float2*)(scores + (i_14 * 2));
        float2 v__16 = make_float2(1.000000e+00f, 1.000000e+00f);
        __9.x = (v__15.x+v__16.x);
        __9.y = (v__15.y+v__16.y);
      *(float2*)(scores + (i_14 * 2)) = __9;
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 32; ++i_15) {
      float2 __10;
        float2 v__17 = *(float2*)(scores + (i_15 * 2));
        float2 v__18 = make_float2(5.000000e-01f, 5.000000e-01f);
        __10.x = (v__17.x*v__18.x);
        __10.y = (v__17.y*v__18.y);
      *(float2*)(scores + (i_15 * 2)) = __10;
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 32; ++i_16) {
      float2 __11;
        float2 v__19 = *(float2*)(acc_o + (i_16 * 2));
        float2 v__20 = make_float2(o_scale[(i_16 & 1)], o_scale[(i_16 & 1)]);
        __11.x = (v__19.x*v__20.x);
        __11.y = (v__19.y*v__20.y);
      *(float2*)(acc_o + (i_16 * 2)) = __11;
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 32; ++i_17) {
      uint1 __12;
      float2 v__21 = *(float2*)(scores + (i_17 * 2));
      ((half2*)(&(__12.x)))->x = (half_t)(v__21.x);
      ((half2*)(&(__12.x)))->y = (half_t)(v__21.y);
      *(uint1*)(acc_s_cast + (i_17 * 2)) = __12;
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[11], 0);
    tl::gemm_rs<128, 128, 128, 8, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[32768])), (&(acc_o[0])));
    #pragma unroll
    for (int i_18 = 0; i_18 < 32; ++i_18) {
      uint1 __13;
      float2 v__22 = *(float2*)(acc_o + (i_18 * 2));
      ((half2*)(&(__13.x)))->x = (half_t)(v__22.x);
      ((half2*)(&(__13.x)))->y = (half_t)(v__22.y);
      *(uint1*)(Output + (((((((((int)blockIdx.x) * 524288) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_18 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __13;
    }
  }
}

