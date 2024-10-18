import re

def replace(code):
    matches = [m.start() for m in re.finditer(r'tl::fence_proxy_async\(\);', code)]
    
    for i in range(1, 5):
        if i < len(matches):
            start_pos = matches[i]
            end_pos = start_pos + len('tl::fence_proxy_async();')
            code = code[:start_pos] + "//" + code[start_pos:]
            matches = [m + 2 for m in matches]

    return code