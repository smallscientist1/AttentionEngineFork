import re

def replace(code):
    gemm_pattern = r"(tl::gemm_rs<[^>]+>)"
    mbarrier_pattern = r"tl::mbarrier_arrive\([^;]+\);"
    
    def add_neg1(match):
        template = match.group(0)
        return template[:-1] + ", -1>"
    
    gemm_match = re.search(gemm_pattern, code)
    mbarrier_match = re.search(mbarrier_pattern, code[gemm_match.end():])
    code = re.sub(gemm_pattern, add_neg1, code)
    # code = re.sub("tl::mbarrier_arrive", "// tl::mbarrier_arrive", code[gemm_match.end():])
    modified_code = code[:gemm_match.end()] + re.sub("tl::mbarrier_arrive", "// tl::mbarrier_arrive", code[gemm_match.end():])

    assert gemm_match and mbarrier_match
    # if gemm_match:
    #     print("gemm_match:", gemm_match.group(0))
    # if mbarrier_match:
    #     print("mbarrier_match:", mbarrier_match.group(0))

    def find_for_loop_bounds(code_string):
        for_loop_pattern = r"for\s*\(\s*int\s*k_1\s*=\s*0\s*;\s*k_1\s*<\s*\d+\s*;\s*\+\+k_1\s*\)\s*\{"
        for_loop_start_match = re.search(for_loop_pattern, code_string)
        
        if for_loop_start_match:
            loop_start_pos = for_loop_start_match.start()
            open_braces = 0
            for i in range(for_loop_start_match.end(), len(code_string)):
                if code_string[i] == '{':
                    open_braces += 1
                elif code_string[i] == '}':
                    if open_braces == 0:
                        # Found the closing brace for the loop
                        loop_end_pos = i + 1
                        return code_string[for_loop_start_match.start():loop_end_pos], loop_start_pos, loop_end_pos
                    else:
                        open_braces -= 1
        return None

    # print("modified_code:", modified_code)
    for_loop, loop_start_pos, loop_end_pos = find_for_loop_bounds(modified_code)

    def add_sync_in_loop(loop_code, _arrive_code):
        def replace_k1_with_k1_minus_1(code_string):
            _code_string = re.sub(r"\bk_1\b", "(k_1 - 1)", code_string)
            return _code_string

        arrive_code = replace_k1_with_k1_minus_1(_arrive_code)
        sync_code = "if (k_1 > 0) {\ncute::warpgroup_wait<0>();\n" + arrive_code + "\n}\n"

        def find_sync_point(code_string):
            b_dequantize_pattern = r"#pragma unroll\s*for\s*\(int\s*\w+\s*=\s*0\s*;\s*\w+\s*<\s*\d+\s*;\s*\+\+\w+\s*\)\s*\{[^}]+B_dequantize_prev_local[^}]+\}"
            b_dequantize_match = re.search(b_dequantize_pattern, code_string)
            # print("b_dequantize_match:", b_dequantize_match.group(0))
            assert b_dequantize_match
            loop_start_pos = b_dequantize_match.start()
            return loop_start_pos

        loop_start_pos = find_sync_point(loop_code)
        code = loop_code[:loop_start_pos] + sync_code + loop_code[loop_start_pos:]
        return code

    # print(find_for_loop_bounds(modified_code))
    modified_loop = add_sync_in_loop(for_loop, mbarrier_match.group(0))
    modified_code = modified_code[:loop_start_pos] + modified_loop + "\ncute::warpgroup_wait<0>();" + modified_code[loop_end_pos:]
    
    return modified_code

# code = """
# #include <tl_templates/gemm.h>
# #include <tl_templates/copy.h>
# #include <tl_templates/reduce.h>
# #include <tl_templates/ldsm.h>
# #include <tl_templates/threadblock_swizzle.h>

# extern "C" __global__ void __launch_bounds__(384) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap Ct_desc) {
#   extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
#   float Ct_local[128];
#   uchar B_local[16];
#   half_t B_dequantize_local[32];
#   half_t B_dequantize_prev_local[32];
#   __shared__ uint64_t _mbarrier[10];
#   if (((int)threadIdx.x) == 0) {
#     tl::prefetch_tma_descriptor(A_desc);
#     tl::prefetch_tma_descriptor(B_desc);
#     tl::prefetch_tma_descriptor(Ct_desc);
#     tl::mbarrier_init(_mbarrier[0], 128);
#     tl::mbarrier_init(_mbarrier[1], 128);
#     tl::mbarrier_init(_mbarrier[2], 128);
#     tl::mbarrier_init(_mbarrier[3], 128);
#     tl::mbarrier_init(_mbarrier[4], 256);
#     tl::mbarrier_init(_mbarrier[5], 256);
#     tl::mbarrier_init(_mbarrier[6], 256);
#     tl::mbarrier_init(_mbarrier[7], 256);
#     tl::mbarrier_init(_mbarrier[8], 256);
#     tl::mbarrier_init(_mbarrier[9], 256);
#   }
#   __syncthreads();
#   if (256 <= ((int)threadIdx.x)) {
#     tl::warpgroup_reg_dealloc<24>();
#     for (int k = 0; k < 128; ++k) {
#       tl::mbarrier_wait(_mbarrier[((k & 3) + 4)], (((k & 7) >> 2) ^ 1));
#       if (((int)threadIdx.x) == 256) {
#         tl::mbarrier_expect_tx(_mbarrier[(k & 3)], 32768);
#       }
#       if (((int)threadIdx.x) == 256) {
#         tl::tma_load(A_desc, _mbarrier[(k & 3)], (&(((half_t*)buf_dyn_shmem)[(((k & 3) * 16384) + 8192)])), (k * 64), (((int)blockIdx.y) * 256));
#       }
#       if (((int)threadIdx.x) == 256) {
#         tl::mbarrier_expect_tx(_mbarrier[(k & 3)], 4096);
#       }
#       if (((int)threadIdx.x) == 256) {
#         tl::tma_load(B_desc, _mbarrier[(k & 3)], (&(buf_dyn_shmem[((k & 3) * 4096)])), (k * 32), (((int)blockIdx.x) * 128));
#       }
#       tl::mbarrier_arrive(_mbarrier[(k & 3)]);
#     }
#   } else {
#     tl::warpgroup_reg_alloc<240>();
#     #pragma unroll
#     for (int i = 0; i < 64; ++i) {
#       *(float2*)(Ct_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
#     }
#     for (int k_1 = 0; k_1 < 128; ++k_1) {
#       tl::mbarrier_wait(_mbarrier[(k_1 & 3)], ((k_1 & 7) >> 2));
#       #pragma unroll
#       for (int i_1 = 0; i_1 < 16; ++i_1) {
#         B_local[i_1] = buf_dyn_shmem[(((((((k_1 & 3) * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_1 >> 3) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_1 & 7) * 4)) + (((int)threadIdx.x) & 3))];
#       }
#       #pragma unroll
#       for (int i_2 = 0; i_2 < 16; ++i_2) {
#           ushort2 __1;
#             ushort2 __2;
#               ushort2 __3;
#                 ushort2 __4;
#                   ushort2 __5;
#                     ushort2 __6;
#                       ushort2 v_ = make_ushort2(((ushort)B_local[(((i_2 & 1) * 8) + (i_2 >> 1))]), ((ushort)B_local[(((i_2 & 1) * 8) + (i_2 >> 1))]));
#                       ushort2 v__1 = make_ushort2(((ushort)0)+((ushort)4*0), ((ushort)0)+((ushort)4*1));
#                       __6.x = (v_.x >> v__1.x);
#                       __6.y = (v_.y >> v__1.y);
#                     ushort2 v__2 = make_ushort2((ushort)15, (ushort)15);
#                     __5.x = (__6.x & v__2.x);
#                     __5.y = (__6.y & v__2.y);
#                   ushort2 v__3 = make_ushort2((ushort)7, (ushort)7);
#                   __4.x = (__5.x & v__3.x);
#                   __4.y = (__5.y & v__3.y);
#                 ushort2 v__4 = make_ushort2((ushort)8, (ushort)8);
#                 __3.x = (__4.x | v__4.x);
#                 __3.y = (__4.y | v__4.y);
#               ushort2 __7;
#                 ushort2 __8;
#                   ushort2 __9;
#                     ushort2 __10;
#                       ushort2 v__5 = make_ushort2(((ushort)B_local[(((i_2 & 1) * 8) + (i_2 >> 1))]), ((ushort)B_local[(((i_2 & 1) * 8) + (i_2 >> 1))]));
#                       ushort2 v__6 = make_ushort2(((ushort)0)+((ushort)4*0), ((ushort)0)+((ushort)4*1));
#                       __10.x = (v__5.x >> v__6.x);
#                       __10.y = (v__5.y >> v__6.y);
#                     ushort2 v__7 = make_ushort2((ushort)15, (ushort)15);
#                     __9.x = (__10.x & v__7.x);
#                     __9.y = (__10.y & v__7.y);
#                   ushort2 v__8 = make_ushort2((ushort)3, (ushort)3);
#                   __8.x = (__9.x >> v__8.x);
#                   __8.y = (__9.y >> v__8.y);
#                 ushort2 v__9 = make_ushort2((ushort)5, (ushort)5);
#                 __7.x = (__8.x << v__9.x);
#                 __7.y = (__8.y << v__9.y);
#               __2.x = (__3.x | __7.x);
#               __2.y = (__3.y | __7.y);
#             ushort2 v__10 = make_ushort2((ushort)10, (ushort)10);
#             __1.x = (__2.x << v__10.x);
#             __1.y = (__2.y << v__10.y);
#         *(uint1*)(B_dequantize_local + (i_2 * 2)) = (*(uint1 *)(&(__1)));
#       }
#       #pragma unroll
#       for (int i_3 = 0; i_3 < 16; ++i_3) {
#         *(uint1*)(B_dequantize_prev_local + (i_3 * 2)) = *(uint1*)(B_dequantize_local + (i_3 * 2));
#       }
#       tl::fence_proxy_async();
#       tl::gemm_rs<128, 256, 64, 8, 1, 0, 1>((&(B_dequantize_prev_local[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 3) * 16384) + 8192)])), (&(Ct_local[0])));
#       tl::mbarrier_arrive(_mbarrier[((k_1 & 3) + 4)]);
#     }
#     tl::syncthreads_partial(_mbarrier[8]);
#     #pragma unroll
#     for (int i_4 = 0; i_4 < 16; ++i_4) {
#       tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((i_4 >> 2) * 8192) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), __pack_half2(((half_t)Ct_local[(i_4 * 8)]), ((half_t)Ct_local[((i_4 * 8) + 1)])), __pack_half2(((half_t)Ct_local[((i_4 * 8) + 2)]), ((half_t)Ct_local[((i_4 * 8) + 3)])), __pack_half2(((half_t)Ct_local[((i_4 * 8) + 4)]), ((half_t)Ct_local[((i_4 * 8) + 5)])), __pack_half2(((half_t)Ct_local[((i_4 * 8) + 6)]), ((half_t)Ct_local[((i_4 * 8) + 7)])));
#     }
#     tl::fence_proxy_async();
#     tl::syncthreads_partial(_mbarrier[9]);
#     if (((int)threadIdx.x) == 0) {
#       tl::tma_store(Ct_desc, (&(((half_t*)buf_dyn_shmem)[8192])), (((int)blockIdx.y) * 256), (((int)blockIdx.x) * 128));
#       tl::tma_store(Ct_desc, (&(((half_t*)buf_dyn_shmem)[16384])), ((((int)blockIdx.y) * 256) + 64), (((int)blockIdx.x) * 128));
#       tl::tma_store(Ct_desc, (&(((half_t*)buf_dyn_shmem)[24576])), ((((int)blockIdx.y) * 256) + 128), (((int)blockIdx.x) * 128));
#       tl::tma_store(Ct_desc, (&(((half_t*)buf_dyn_shmem)[32768])), ((((int)blockIdx.y) * 256) + 192), (((int)blockIdx.x) * 128));
#     }
#   }
# }
# """

# # replace(code)
# print(replace(code))