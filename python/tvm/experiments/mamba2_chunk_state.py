import re

def replace(code):
    gemm_pattern = r"(tl::gemm_rs<[^>]+>)"
    mbarrier_pattern = r"tl::mbarrier_arrive\([^;]+\);"
    
    def add_neg1(match):
        template = match.group(0)
        return template[:-1] + ", -1>"
    
    matches = [m.start() for m in re.finditer(r'tl::fence_proxy_async\(\);', code)]
    
    for i in range(0, 2):
        if i < len(matches):
            start_pos = matches[i]
            end_pos = start_pos + len('tl::fence_proxy_async();')
            code = code[:start_pos] + "//" + code[start_pos:]
            matches = [m + 2 for m in matches]

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
            b_dequantize_pattern = r"#pragma unroll\s*for\s*\(int\s*\w+\s*=\s*0\s*;\s*\w+\s*<\s*\d+\s*;\s*\+\+\w+\s*\)\s*\{[^}]+scale[^}]+\}"
            scale_match = re.search(b_dequantize_pattern, code_string)
            # print("scale_match:", scale_match.group(0))
            assert scale_match
            loop_start_pos = scale_match.start()
            return loop_start_pos

        loop_start_pos = find_sync_point(loop_code)
        code = loop_code[:loop_start_pos] + sync_code + loop_code[loop_start_pos:]
        return code

    # print(find_for_loop_bounds(modified_code))
    modified_loop = add_sync_in_loop(for_loop, mbarrier_match.group(0))
    modified_code = modified_code[:loop_start_pos] + modified_loop + "\ncute::warpgroup_wait<0>();" + modified_code[loop_end_pos:]
    
    return modified_code