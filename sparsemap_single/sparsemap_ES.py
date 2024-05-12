import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from parse_timeloop_output import parse_timeloop_stats
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader
import yaml
import os
import inspect
from copy import deepcopy
import shutil, argparse, pprint, sys, subprocess
import random
from datetime import datetime
import csv

#五层存储的ES  此版本暂时只考虑mapping

OVERWRITE = 1

dimensions = {'C': 3, 'M': 96, 'N': 4, 'P': 54, 'Q': 54, 'R': 12, 'S': 12}

N_GENERATIONS = 200
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation

OBJECT = 'cycles'       #优化目标
'''
    'problem': problem,
    'utilization': arithmetic_utilization,
    'cycles': max_cycles,
    'energy_pJ': energy_pJ,
    'energy_per_mac': energy_pJ/macs,
    'macs': macs,
    'energy_breakdown_pJ': energy_breakdown_pJ,
    'bandwidth_and_cycles': bandwidth_and_cycles

    '''

##质因数分解
def prime_factorization(n):
    factors = []
    divisor = 2

    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1

    return factors

cf = prime_factorization(dimensions['C'])
mf = prime_factorization(dimensions['M'])
nf = prime_factorization(dimensions['N'])
pf = prime_factorization(dimensions['P'])
qf = prime_factorization(dimensions['Q'])
rf = prime_factorization(dimensions['R'])
sf = prime_factorization(dimensions['S'])
factor_list = [cf,mf,nf,pf,qf,rf,sf]                    #质因数汇总

lc = len(cf)
lm = len(mf)
lr = len(rf)
ls = len(sf)
ln = len(nf)
lp = len(pf)
lq = len(qf)
len_list = [lc,lm,ln,lp,lq,lr,ls]                       #每个维度的质因数长度

mapping_encoding_len = lc+lm+lr+ls+ln+lp+lq

def map_encode(size):
    size_code = []
    for dim_index in range(7):
        factors_of_dim = factor_list[dim_index]
        for factor in factors_of_dim:
            for i in range(5):
                if size[i][dim_index]%factor == 0:
                    size[i][dim_index] = size[i][dim_index]//factor
                    size_code.append(i)
    return size_code

def map_decode(code):
    size = [[1,1,1,1,1,1,1]for _ in range(5)]
    start = 0
    for i in range(7):
        for j in range(len_list[i]):
            size[code[start+j]][i] *= factor_list[i][j]
        start = start + len_list[i]
    return size

def cantor_encode(permutation):
    n = len(permutation)
    encoded = 0
    factorial = math.factorial(n-1)
    for i in range(n):
        count = 0
        for j in range(i+1, n):
            if permutation[j] < permutation[i]:
                count += 1
        encoded += count * factorial
        if i < n-1:
            factorial //= (n-i-1)
    return encoded + 1  # 加一是因为康托展开从1开始计数

def cantor_decode(encoded, n):
    permutation = []
    factorial = math.factorial(n-1)
    encoded -= 1  # 康托展开从1开始计数，这里要减一
    available = list(range(n))
    for i in range(n):
        index = encoded // factorial
        encoded %= factorial
        permutation.append(available[index])
        available.pop(index)
        if i < n-1:
            factorial //= (n-i-1)
    return permutation

#run_timeloop负责调用sparseloopp
def run_timeloop(job_name, input_dict, base_dir, ert_path, art_path):
    output_dir = os.path.join(base_dir +"/outputs")
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not OVERWRITE:
            print("Found existing results: ", output_dir)
            return
        else:
            print("Found and overwrite existing results: ", output_dir)

    # reuse generated ERT and ART files
    shutil.copy(ert_path, os.path.join(base_dir, "ERT.yaml"))
    shutil.copy(art_path, os.path.join(base_dir, "ART.yaml"))
    
    input_file_path = os.path.join(base_dir, "aggregated_input.yaml")
    #ert_file_path = os.path.join(base_dir, "ERT.yaml")
    #art_file_path = os.path.join(base_dir, "ART.yaml")
    #logfile_path = os.path.join(output_dir, "timeloop.log")
    
    yaml.dump(input_dict, open(input_file_path, "w"), default_flow_style=False)
    os.chdir(output_dir)
    subprocess_cmd = ["timeloop-model", input_file_path, ert_path, art_path]
    #subprocess_cmd = ["timeloop-mapper", input_file_path, ert_path, art_path]
    print("\tRunning test: ", job_name)

    p = subprocess.Popen(subprocess_cmd)
    try:
        p.communicate(timeout=0.2) 
    except subprocess.TimeoutExpired:
        p.terminate()
        #this_file_path = '/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/MCTS_initialize.py'
        #this_directory = os.path.dirname(this_file_path)
        #base_output_dir = os.path.join(this_directory, "MCTS_search_outputs")
        #output_file_path = os.path.join(base_output_dir, job_name, 'outputs', "timeloop-model.map+stats.xml")
        path = '/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/MCTS_search_outputs/MCTS_searching/outputs/timeloop-model.map+stats.xml'
        if os.path.exists(path):
            os.remove(path)
        return 

#evaluate  输入一个individual,返回优化目标的评估值  这个版本暂时只考虑maping  individual是numpy       待修改
def evaluate(individal):                                    

    dimension = ['C','M','N','P','Q','R','S']
    
    #----------------------------------------------------  解码mapping  ------------------------------------------------------
    
    mapping = map_decode(individal[83:83+mapping_encoding_len].tolist())

    

    #print(dimention_factors)
    split_DRAM_to_GB = individal[74]
    split_GB_to_PEB = individal[75]

    permutation_DRAM_T_order = cantor_decode(individal[78],7)
    #print("per = ",permutation_DRAM_T_order)
    permutation_GB_T_order = cantor_decode(individal[79],7)
    #print("per = ",permutation_GB_T_order)
    permutation_GB_S_order = cantor_decode(individal[80],7)
    permutation_PEB_T_order = cantor_decode(individal[81],7)
    permutation_PEB_S_order = cantor_decode(individal[82],7)

    permutation_DRAM_T_str = (dimension[permutation_DRAM_T_order[0]] + dimension[permutation_DRAM_T_order[1]] + dimension[permutation_DRAM_T_order[2]]
                  + dimension[permutation_DRAM_T_order[3]] + dimension[permutation_DRAM_T_order[4]] + dimension[permutation_DRAM_T_order[5]]
                  + dimension[permutation_DRAM_T_order[6]])


    permutation_GB_T_str = (dimension[permutation_GB_T_order[0]] + dimension[permutation_GB_T_order[1]] + dimension[permutation_GB_T_order[2]]
                  + dimension[permutation_GB_T_order[3]] + dimension[permutation_GB_T_order[4]] + dimension[permutation_GB_T_order[5]]
                  + dimension[permutation_GB_T_order[6]])

    permutation_GB_S_str = (dimension[permutation_GB_S_order[0]] + dimension[permutation_GB_S_order[1]] + dimension[permutation_GB_S_order[2]]
                  + dimension[permutation_GB_S_order[3]] + dimension[permutation_GB_S_order[4]] + dimension[permutation_GB_S_order[5]]
                  + dimension[permutation_GB_S_order[6]])

    permutation_PEB_T_str = (dimension[permutation_PEB_T_order[0]] + dimension[permutation_PEB_T_order[1]] + dimension[permutation_PEB_T_order[2]]
                  + dimension[permutation_PEB_T_order[3]] + dimension[permutation_PEB_T_order[4]] + dimension[permutation_PEB_T_order[5]]
                  + dimension[permutation_PEB_T_order[6]])
    
    permutation_PEB_S_str = (dimension[permutation_PEB_S_order[0]] + dimension[permutation_PEB_S_order[1]] + dimension[permutation_PEB_S_order[2]]
                  + dimension[permutation_PEB_S_order[3]] + dimension[permutation_PEB_S_order[4]] + dimension[permutation_PEB_S_order[5]]
                  + dimension[permutation_PEB_S_order[6]])


    dimention_factors = [
        {
        'C': mapping[0][0],
        'M': mapping[0][1],
        'N': mapping[0][2],
        'P': mapping[0][3],
        'Q': mapping[0][4],
        'R': mapping[0][5],
        'S': mapping[0][6]
        },
        {
        'C': mapping[1][0],
        'M': mapping[1][1],
        'N': mapping[1][2],
        'P': mapping[1][3],
        'Q': mapping[1][4],
        'R': mapping[1][5],
        'S': mapping[1][6]
        },
        {
        'C': mapping[2][0],
        'M': mapping[2][1],
        'N': mapping[2][2],
        'P': mapping[2][3],
        'Q': mapping[2][4],
        'R': mapping[2][5],
        'S': mapping[2][6]
        },
        {
        'C': mapping[3][0],
        'M': mapping[3][1],
        'N': mapping[3][2],
        'P': mapping[3][3],
        'Q': mapping[3][4],
        'R': mapping[3][5],
        'S': mapping[3][6]
        },
        {
        'C': mapping[4][0],
        'M': mapping[4][1],
        'N': mapping[4][2],
        'P': mapping[4][3],
        'Q': mapping[4][4],
        'R': mapping[4][5],
        'S': mapping[4][6]
        }
        
    ]

    factors_DRAM_T_str = ('C='+str(dimention_factors[0]['C'])+' '+
                        'M='+str(dimention_factors[0]['M'])+' '+
                        'N='+str(dimention_factors[0]['N'])+' '+ 
                        'P='+str(dimention_factors[0]['P'])+' '+
                        'Q='+str(dimention_factors[0]['Q'])+' '+
                        'R='+str(dimention_factors[0]['R'])+' '+
                        'S='+str(dimention_factors[0]['S']))

    factors_GB_T_str = ('C='+str(dimention_factors[1]['C'])+' '+
                        'M='+str(dimention_factors[1]['M'])+' '+
                        'N='+str(dimention_factors[1]['N'])+' '+ 
                        'P='+str(dimention_factors[1]['P'])+' '+
                        'Q='+str(dimention_factors[1]['Q'])+' '+
                        'R='+str(dimention_factors[1]['R'])+' '+
                        'S='+str(dimention_factors[1]['S']))

    factors_GB_S_str = ('C='+str(dimention_factors[2]['C'])+' '+
                        'M='+str(dimention_factors[2]['M'])+' '+
                        'N='+str(dimention_factors[2]['N'])+' '+ 
                        'P='+str(dimention_factors[2]['P'])+' '+
                        'Q='+str(dimention_factors[2]['Q'])+' '+
                        'R='+str(dimention_factors[2]['R'])+' '+
                        'S='+str(dimention_factors[2]['S']))
    
    factors_PEB_T_str = ('C='+str(dimention_factors[3]['C'])+' '+
                        'M='+str(dimention_factors[3]['M'])+' '+
                        'N='+str(dimention_factors[3]['N'])+' '+ 
                        'P='+str(dimention_factors[3]['P'])+' '+
                        'Q='+str(dimention_factors[3]['Q'])+' '+
                        'R='+str(dimention_factors[3]['R'])+' '+
                        'S='+str(dimention_factors[3]['S']))
    
    factors_PEB_S_str = ('C='+str(dimention_factors[4]['C'])+' '+
                        'M='+str(dimention_factors[4]['M'])+' '+
                        'N='+str(dimention_factors[4]['N'])+' '+ 
                        'P='+str(dimention_factors[4]['P'])+' '+
                        'Q='+str(dimention_factors[4]['Q'])+' '+
                        'R='+str(dimention_factors[4]['R'])+' '+
                        'S='+str(dimention_factors[4]['S']))

    
    bypass_GB = individal[76]
    bypass_PEB = individal[77]

    #iwo = ['Weights','Inputs','Outputs']
    iwo_GB = []
    a = bypass_GB//4
    if a == 1:
        iwo_GB.append('Inputs')
    bypass_GB = bypass_GB%4
    b = bypass_GB//2
    if b == 1:
        iwo_GB.append('Weight')
    bypass_GB = bypass_GB%2
    if bypass_GB == 1:
        iwo_GB.append('Outputs')

    iwo_PEB = []
    a = bypass_PEB//4
    if a == 1:
        iwo_PEB.append('Inputs')
    bypass_PEB = bypass_PEB%4
    b = bypass_PEB//2
    if b == 1:
        iwo_PEB.append('Weight')
    bypass_PEB = bypass_PEB%2
    if bypass_PEB == 1:
        iwo_PEB.append('Outputs')



    mapping_variables = {
    'permutation_DRAM_T': permutation_DRAM_T_str,
    'factors_DRAM_T': factors_DRAM_T_str,

    'permutation_GB_T': permutation_GB_T_str,
    'factors_GB_T': factors_GB_T_str,

    'permutation_GB_S': permutation_GB_S_str,
    'split_DRAM_to_GB': split_DRAM_to_GB,
    'factors_GB_S': factors_GB_S_str,

    'permutation_PEB_T': permutation_PEB_T_str,
    'factors_PEB_T': factors_PEB_T_str,

    'permutation_PEB_S': permutation_PEB_S_str,
    'split_GB_to_PEB': split_GB_to_PEB,
    'factors_PEB_S': factors_PEB_S_str,

    'iwo_GB':iwo_GB,
    'iwo_PEB':iwo_PEB

    }

    # ------------------------------------  mapping  输入文件生成  -------------------------------------------
    # 读取模板文件
    #print(os.getcwd())
    
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)
    print("this_direc = ",this_directory)
    #os.chdir(this_directory)
    #print(os.getcwd())
    with open('yamls/mapping_conv.j2', 'r') as template_file_mapping:
        template_content_mapping = template_file_mapping.read()

    # 创建 Jinja2 模板对象
    template_mapping = Template(template_content_mapping)

    # 渲染模板并生成 YAML 内容
    rendered_yaml_mapping = template_mapping.render(mapping_variables)

    # 将生成的 YAML 内容写入文件
    with open('yamls/mapping_conv_output_MCTS.yaml', 'w') as output_file_mapping:
        output_file_mapping.write(rendered_yaml_mapping)

#----------------------------------------------调用sparse_loop-------------------------------------------------------

    
    
    problem_template_path = os.path.join(this_directory,"yamls", "workload_example.yaml")
    arch_path = os.path.join(this_directory, "yamls","arch_edge.yaml")
    #component_path = os.path.join(this_directory, "..", "multiSCNN","fig13_dstc_setup","input_specs",  "compound_components.yaml")
    mapping_path = os.path.join(this_directory,"yamls", "mapping_conv_output_MCTS.yaml")
    #mapper_path = os.path.join(this_directory, "mapper.yaml")
    #sparse_opt_path = os.path.join(this_directory, "sparse_opt_output.yaml")
    #sparse_opt_path = os.path.join(this_directory, "..", "multiSCNN","single_core_optimization",  "yaml_gen", "sparse_opt_output.yaml")



    ert_path = os.path.join(this_directory,"..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ERT.yaml")
    art_path = os.path.join(this_directory, "..","..", "evaluation_setups","fig13_dstc_setup","input_specs",  "ART.yaml")
    #print(ert_path)
    
    # load all yaml input specs
    problem_template = yaml.load(open(problem_template_path), Loader = yaml.SafeLoader)
    arch = yaml.load(open(arch_path), Loader = yaml.SafeLoader)
    #components = yaml.load(open(component_path), Loader = yaml.SafeLoader)
    mapping = yaml.load(open(mapping_path), Loader = yaml.SafeLoader)
    #mapper = yaml.load(open(mapper_path), Loader = yaml.SafeLoader)
    #sparse_opt = yaml.load(open(sparse_opt_path), Loader = yaml.SafeLoader)
    
    output_base_dir = os.path.join(this_directory, "MCTS_search_outputs")

    new_problem = deepcopy(problem_template)

    #new_problem["problem"]["instance"]["densities"]["A"]["density"] = 0.55
    #new_problem["problem"]["instance"]["densities"]["B"]["density"] = 0.35

    aggregated_input = {}
    aggregated_input.update(arch)
    aggregated_input.update(new_problem)
    #aggregated_input.update(components)
    aggregated_input.update(mapping)
    #aggregated_input.update(mapper)
    #aggregated_input.update(sparse_opt)
    
    
    job_name  = "MCTS_searching"
    

#---------------------------------------------------调用,生成output---------------------------------------------------------
    run_timeloop(job_name, aggregated_input, os.path.join(output_base_dir, job_name), ert_path, art_path)


#------------------------------------------------------读取数据--------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++本实验优化目标+++++++++++++++++++++++++++++++++++++++++
    stat_type = OBJECT
    '''
    'problem': problem,
    'utilization': arithmetic_utilization,
    'cycles': max_cycles,
    'energy_pJ': energy_pJ,
    'energy_per_mac': energy_pJ/macs,
    'macs': macs,
    'energy_breakdown_pJ': energy_breakdown_pJ,
    'bandwidth_and_cycles': bandwidth_and_cycles

    '''
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    base_output_dir = os.path.join(this_directory, "MCTS_search_outputs")
    output_file_path = os.path.join(base_output_dir, job_name, 'outputs', "timeloop-model.map+stats.xml")
    if os.path.exists(output_file_path):
        job_output_stats = parse_timeloop_stats(output_file_path)
        os.remove(output_file_path)
        fitness = job_output_stats[stat_type]
    else:
        fitness = 10000000000000000000
    return fitness

DNA_SIZE = 83+mapping_encoding_len            # DNA 

'''
[0,29]          Input 压缩格式           uop---0    b---1    cp---2    rle---3
[30,49]         Weight 压缩格式          uop---0    b---1    cp---2    rle---3
[50,69]         Output 压缩格式          uop---0    b---1    cp---2    rle---3
[70,71]         GlobleBuffer 中的 skip/gate
[72,73]            PE_Buffer 中的 skip/gate
[74,75]                    split 位置
[76,77]                    bypass_choice
[78,82]                    cantor编码的permutation
[83,83+len_list+1]         map_size  编码

'''
def roulette_wheel_selection(numbers):        #输入适应度表，输出一个随概率选择的索引号
    # 计算总的适应度值（这里假设适应度值就是数字本身）
    total_fitness = sum(numbers)
    # 计算每个数字的选择概率（适应度值越高，被选中的概率越大）
    probabilities = [num / total_fitness for num in numbers]

    # 生成一个随机概率值
    random_prob = random.random()
    cumulative_prob = 0
    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if random_prob <= cumulative_prob:
            selected_idx = i
            break     
    return selected_idx

def population_initialize(inin_path):
    
    pop_list = []
    # 创建一个长度为10的空列表来存储每行第一个数
    first_numbers_devided_by1 = []
    with open(inin_path, mode='r') as file:
        reader = csv.reader(file)
        # 逐行读取CSV文件
        for row in reader:
            # 获取每行的第一个数，并将其添加到first_numbers列表中
            first_number = int(row[0])  # 假设第一个数是整数
            first_numbers_devided_by1.append(1/first_number)
        for i in range(POP_SIZE):
            indi = [0 for m in range(POP_SIZE)]
            row_idx = roulette_wheel_selection(first_numbers_devided_by1)
            for _ in range(row_idx-1):
                next(reader)
            target_row = next(reader)
            data_list = target_row[1:]
            indi[70:DNA_SIZE] = data_list
            pop_list.append(indi)
    population = np.array(pop_list)

    return population


def select_order(population, fitness):
    # 根据适应度对个体进行排序的索引
    sorted_indices = np.argsort(fitness)[::-1]

    # 选择排名前n的个体及其适应度
    selected_population = population[sorted_indices[:POP_SIZE - N_KID]]
    selected_fitness = fitness[sorted_indices[:POP_SIZE - N_KID]]

    return selected_population, selected_fitness


def crossover(parents, n_kid):
    kids = []
    for i in range(n_kid):
        selected_indices = np.random.choice(parents.shape[0], size=2, replace=False)
        parent1 = parents[selected_indices[0]]
        parent2 = parents[selected_indices[1]]
        crossover_point = np.random.randint(79, 83+mapping_encoding_len-1)                               #决定染色体交叉点
        kid = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        kids.append(kid)
    return np.array(kids)

def mutate(parents, n_kid):                    #暂时只用mutate产生子代
    kids = []
    for i in range(n_kid):
        selected_indices = np.random.choice(parents.shape[0], size=1, replace=False)
        parent = parents[selected_indices]
        kid = parent.copy()
        mutate_choice = np.random.randint(0,2)
        if mutate_choice == 0:
            cantor_mutate_position = np.random.randint(78,83)        #康托展开变异位置
            p_or_m = np.random.randint(0,2)               #加还是减
            if p_or_m == 0:
                kid[cantor_mutate_position] = (kid[cantor_mutate_position]+1)//5040
            else:
                kid[cantor_mutate_position] = (kid[cantor_mutate_position]-1)//5040
        else:
            mapsize_mutate_position = np.random.randint(83,83+mapping_encoding_len)        #mapsize变异位置
            p_or_m = np.random.randint(0,2)               #加还是减
            if p_or_m == 0:
                kid[mapsize_mutate_position] = (kid[cantor_mutate_position]+1)//5040
            else:
                kid[mapsize_mutate_position] = (kid[cantor_mutate_position]-1)//5040

        kids.append(kid)
    return np.array(kids)

def envolve(population,fitness):                                     
    selected_population, parents_fitness = select_order(population, fitness)   #从上一代中挑出50个
    kids = mutate(selected_population, N_KID)                
    kids_fitness = np.array([evaluate(indi) for indi in kids])                                                    #生成子代       
    next_generation_candidates = np.concatenate([selected_population, kids])  
    fitness = np.append(parents_fitness,kids_fitness)
    return next_generation_candidates, fitness


def main():
    initialize_path = reading_path = "/home/workspace/2022.micro.artifact/multiSCNN/sparsemap_single/outputs/example/outputs/initialize_mappings_found.csv"
    population = population_initialize()
    fitness = np.array([evaluate(indi) for indi in population])
    generation_best_individual = []
    generation_best_performance = []

    for i in range(N_GENERATIONS):
        population,fitness = envolve(population,fitness)
        best_indi_index = np.argmax(fitness)
        generation_best_individual.append(population[best_indi_index,:].tolist())
        generation_best_performance.append(fitness[best_indi_index])
        print("=====================================================================\n")
        print("GENERATION = ",i)
        print("Best performance is",fitness[best_indi_index])
        print("=====================================================================\n")
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    np.savetxt(formatted_now +'_best_individual.txt', generation_best_individual)
    np.savetxt(formatted_now +'_best_performance.txt', generation_best_performance)


if __name__ == "__main__":
    main()