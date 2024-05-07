import yaml

reading_path = "/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/outputs/example/outputs/timeloop-mapper.map.yaml"


def convert_yaml_to_state(yaml_path):

    state = {}
    with open(yaml_path, 'r') as file:
        mapping_data = yaml.safe_load(file)

    arr = [[0 for _ in range(7)] for _ in range(5)]
    perm = [[] for _ in range(5)]
    sp = [0,0]
    pb = [0,0]

    # 定义列索引顺序
    columns = ['C', 'M', 'N', 'P', 'Q', 'R', 'S']

    # 遍历解析后的数据
    for entry in mapping_data['mapping']:
        target = entry['target']
        type = entry['type']
        factors = entry.get('factors', None)
        permutation_values = entry.get('permutation', None)

        
        # 确定填充位置
        if target == 'DRAM' and type == 'temporal':
            row = 0
        elif target == 'GlobelBuffer' and type == 'temporal':
            row = 1
        elif target == 'GlobelBuffer' and type == 'spatial':
            row = 2
            sp[0] = entry['split']
        elif target == 'PE_buffer' and type == 'temporal':
            row = 3
        elif target == 'PE_buffer' and type == 'spatial':
            row = 4
            sp[1] = entry['split']
        elif target == 'GlobelBuffer' and type == 'datatype':
            iwo = entry['bypass']
            if 'Inputs' in iwo:
                pb[0]+=4
            if 'Weights' in iwo:
                pb[0]+=2
            if 'Outputs' in iwo:
                pb[0]+=1
        elif target == 'PE_buffer' and type == 'datatype':
            iwo = entry['bypass']
            if 'Inputs' in iwo:
                pb[1]+=4
            if 'Weights' in iwo:
                pb[1]+=2
            if 'Outputs' in iwo:
                pb[1]+=1
        else:
            continue  # 跳过其他类型的entry
        
        # 提取factors中的值，并填充到对应位置
        if factors:
            factors_values = factors.split()
            for factor_value in factors_values:
                factor = factor_value[0]  # 提取字母，例如'C'
                number = int(factor_value[1:])  # 提取数字，例如3
                if factor in columns:  # 判断字母是否在列索引顺序中
                    col = columns.index(factor)  # 获取字母在列表中的索引作为列索引
                    arr[row][col] = number

        if permutation_values:
            permutation_values = list(permutation_values)
            for perm_value in permutation_values:
                if perm_value in columns:  # 判断字符是否在列索引顺序中
                    perm[row].append(columns.index(perm_value))

    state['permutations'] = perm
    state['bypass_choice'] = pb
    state['array'] = arr
    state['split'] = sp

    return state


s = convert_yaml_to_state(reading_path)
print(s)