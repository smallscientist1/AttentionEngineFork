#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(384) main_kernel(__grid_constant__ const CUtensorMap K_desc, bfloat16_t* __restrict__ Output, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc, __grid_constant__ const CUtensorMap g_mask_desc, float* __restrict__ g_r) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[160];
  float r_wo_clamp[8];
  float r[1];
  float scores[32];
  bfloat16_t mask[32];
  float scores_1[32];
  float scores_0_sum_0[8];
  float r_new[8];
  bfloat16_t accs_cast_1[32];
  float r_o[4];
  __shared__ uint64_t _mbarrier[11];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(g_mask_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 256);
    tl::mbarrier_init(_mbarrier[4], 256);
    tl::mbarrier_init(_mbarrier[5], 256);
    tl::mbarrier_init(_mbarrier[6], 128);
    tl::mbarrier_init(_mbarrier[7], 256);
    tl::mbarrier_init(_mbarrier[8], 256);
    tl::mbarrier_init(_mbarrier[9], 256);
    tl::mbarrier_init(_mbarrier[10], 256);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) == 256) {
      tl::mbarrier_expect_tx(_mbarrier[6], 65536);
    }
    if (((int)threadIdx.x) == 256) {
      tl::tma_load(Q_desc, _mbarrier[6], (&(((bfloat16_t*)buf_dyn_shmem)[69888])), 0, 0, (((int)blockIdx.x) * 128), 0);
      tl::tma_load(Q_desc, _mbarrier[6], (&(((bfloat16_t*)buf_dyn_shmem)[78080])), 64, 0, (((int)blockIdx.x) * 128), 0);
      tl::tma_load(Q_desc, _mbarrier[6], (&(((bfloat16_t*)buf_dyn_shmem)[86272])), 128, 0, (((int)blockIdx.x) * 128), 0);
      tl::tma_load(Q_desc, _mbarrier[6], (&(((bfloat16_t*)buf_dyn_shmem)[94464])), 192, 0, (((int)blockIdx.x) * 128), 0);
    }
    tl::mbarrier_arrive(_mbarrier[6]);
    for (int k = 0; k < ((((int)blockIdx.x) * 2) + 2); ++k) {
      tl::mbarrier_wait(_mbarrier[3], ((k & 1) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[0], 32768);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(K_desc, _mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[16640])), 0, 0, (k * 64), 0);
        tl::tma_load(K_desc, _mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[20736])), 64, 0, (k * 64), 0);
        tl::tma_load(K_desc, _mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[24832])), 128, 0, (k * 64), 0);
        tl::tma_load(K_desc, _mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[28928])), 192, 0, (k * 64), 0);
      }
      tl::mbarrier_arrive(_mbarrier[0]);
      tl::mbarrier_wait(_mbarrier[4], ((k & 1) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[1], 16384);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(g_mask_desc, _mbarrier[1], (&(((bfloat16_t*)buf_dyn_shmem)[256])), (k * 64), (((int)blockIdx.x) * 128), 0, 0);
      }
      tl::mbarrier_arrive(_mbarrier[1]);
      tl::mbarrier_wait(_mbarrier[5], ((k & 1) ^ 1));
      if (((int)threadIdx.x) == 256) {
        tl::mbarrier_expect_tx(_mbarrier[2], 40960);
      }
      if (((int)threadIdx.x) == 256) {
        tl::tma_load(V_desc, _mbarrier[2], (&(((bfloat16_t*)buf_dyn_shmem)[49408])), 0, 0, (k * 64), 0);
        tl::tma_load(V_desc, _mbarrier[2], (&(((bfloat16_t*)buf_dyn_shmem)[53504])), 64, 0, (k * 64), 0);
        tl::tma_load(V_desc, _mbarrier[2], (&(((bfloat16_t*)buf_dyn_shmem)[57600])), 128, 0, (k * 64), 0);
        tl::tma_load(V_desc, _mbarrier[2], (&(((bfloat16_t*)buf_dyn_shmem)[61696])), 192, 0, (k * 64), 0);
        tl::tma_load(V_desc, _mbarrier[2], (&(((bfloat16_t*)buf_dyn_shmem)[65792])), 256, 0, (k * 64), 0);
      }
      tl::mbarrier_arrive(_mbarrier[2]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 80; ++i) {
      *(float2*)(acc_o + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      r_wo_clamp[i_1] = 0.000000e+00f;
    }
    r[0] = 0.000000e+00f;
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[6], 0);
    for (int k_1 = 0; k_1 < ((((int)blockIdx.x) * 2) + 2); ++k_1) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        *(float2*)(scores + (i_2 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[0], (k_1 & 1));
      tl::gemm_ss<128, 64, 256, 4, 2, 0, 1>((&(((bfloat16_t*)buf_dyn_shmem)[69888])), (&(((bfloat16_t*)buf_dyn_shmem)[16640])), (&(scores[0])));
      tl::mbarrier_arrive(_mbarrier[3]);
      tl::mbarrier_wait(_mbarrier[1], (k_1 & 1));
      #pragma unroll
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((i_3 >> 1) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 256)])), (&(mask[(i_3 * 8)])));
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[4]);
      #pragma unroll
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        float2 __1;
          float2 v_ = *(float2*)(scores + (i_4 * 2));
          float2 __2;
          uint1 v__1 = *(uint1*)(mask + (i_4 * 2));
          __2.x = (float)(((nv_bfloat162*)(&(v__1.x)))->x);
          __2.y = (float)(((nv_bfloat162*)(&(v__1.x)))->y);
          __1.x = (v_.x*__2.x);
          __1.y = (v_.y*__2.y);
        *(float2*)(scores + (i_4 * 2)) = __1;
      }
      tl::syncthreads_partial(_mbarrier[7]);
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        for (int vec = 0; vec < 2; ++vec) {
          ((float*)buf_dyn_shmem)[((((((((((((((int)threadIdx.x) >> 7) * 4096) + ((i_5 >> 3) * 2048)) + (((((int)threadIdx.x) & 127) >> 5) * 512)) + ((i_5 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_5 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + vec) + 16512)] = scores[((i_5 * 2) + vec)];
        }
      }
      tl::syncthreads_partial(_mbarrier[8]);
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        *(float4*)(scores_1 + (i_6 * 4)) = *(float4*)(((float*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 16512));
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        scores_0_sum_0[i_7] = 0.000000e+00f;
        #pragma unroll
        for (int rv = 0; rv < 4; ++rv) {
          scores_0_sum_0[i_7] = (scores_0_sum_0[i_7] + max(scores_1[((i_7 * 4) + rv)], (0.000000e+00f - scores_1[((i_7 * 4) + rv)])));
        }
        scores_0_sum_0[i_7] = tl::AllReduce<tl::SumOp, 16, 1>::run(scores_0_sum_0[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 8; ++i_8) {
        r_wo_clamp[i_8] = (r_wo_clamp[i_8] + scores_0_sum_0[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        r_new[i_9] = max(r_wo_clamp[i_9], 1.000000e+00f);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 8; ++i_10) {
        float4 __3;
          float4 v__2 = *(float4*)(scores_1 + (i_10 * 4));
          float4 v__3 = make_float4(r_new[i_10], r_new[i_10], r_new[i_10], r_new[i_10]);
          __3.x = (v__2.x/v__3.x);
          __3.y = (v__2.y/v__3.y);
          __3.z = (v__2.z/v__3.z);
          __3.w = (v__2.w/v__3.w);
        *(float4*)(scores_1 + (i_10 * 4)) = __3;
      }
      if (((((int)threadIdx.x) & 15) >> 3) == 0) {
        r[0] = (r[0] / r_new[(((int)threadIdx.x) & 7)]);
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 8; ++i_11) {
        uint2 __4;
        float4 v__4 = *(float4*)(scores_1 + (i_11 * 4));
        ((nv_bfloat162*)(&(__4.x)))->x = (bfloat16_t)(v__4.x);
        ((nv_bfloat162*)(&(__4.x)))->y = (bfloat16_t)(v__4.y);
        ((nv_bfloat162*)(&(__4.y)))->x = (bfloat16_t)(v__4.z);
        ((nv_bfloat162*)(&(__4.y)))->y = (bfloat16_t)(v__4.w);
        *(uint2*)(accs_cast_1 + (i_11 * 4)) = __4;
      }
      tl::syncthreads_partial(_mbarrier[9]);
      #pragma unroll
      for (int i_12 = 0; i_12 < 8; ++i_12) {
        *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_12 * 1024) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8448)) = *(uint2*)(accs_cast_1 + (i_12 * 4));
      }
      if (((((int)threadIdx.x) & 15) >> 3) == 0) {
        ((float*)buf_dyn_shmem)[(((((int)threadIdx.x) & 7) * 16) + (((int)threadIdx.x) >> 4))] = r[0];
      }
      tl::syncthreads_partial(_mbarrier[10]);
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        r_o[i_13] = ((float*)buf_dyn_shmem)[(((((i_13 >> 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 80; ++i_14) {
        float2 __5;
          float2 v__5 = *(float2*)(acc_o + (i_14 * 2));
          float2 v__6 = make_float2(r_o[(((i_14 / 40) * 2) + (i_14 & 1))], r_o[(((i_14 / 40) * 2) + (i_14 & 1))]);
          __5.x = (v__5.x*v__6.x);
          __5.y = (v__5.y*v__6.y);
        *(float2*)(acc_o + (i_14 * 2)) = __5;
      }
      r[0] = r_new[(((int)threadIdx.x) & 7)];
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[2], (k_1 & 1));
      tl::gemm_ss<128, 320, 64, 4, 2, 0, 0>((&(((bfloat16_t*)buf_dyn_shmem)[8448])), (&(((bfloat16_t*)buf_dyn_shmem)[49408])), (&(acc_o[0])));
      tl::mbarrier_arrive(_mbarrier[5]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 80; ++i_15) {
      uint1 __6;
      float2 v__7 = *(float2*)(acc_o + (i_15 * 2));
      ((nv_bfloat162*)(&(__6.x)))->x = (bfloat16_t)(v__7.x);
      ((nv_bfloat162*)(&(__6.x)))->y = (bfloat16_t)(v__7.y);
      *(uint1*)(Output + ((((((((((int)blockIdx.x) * 40960) + ((i_15 / 40) * 20480)) + (((((int)threadIdx.x) & 127) >> 5) * 5120)) + ((i_15 & 1) * 2560)) + (((((int)threadIdx.x) & 31) >> 2) * 320)) + ((((int)threadIdx.x) >> 7) * 160)) + (((i_15 % 40) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __6;
    }
    if (((((int)threadIdx.x) & 15) >> 3) == 0) {
      g_r[(((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) & 7) * 16)) + (((int)threadIdx.x) >> 4))] = r[0];
    }
  }
}

