import os
import random

#stem base directory
cur = "c:/Users/COINSE/Downloads/simfl-extension"
os.chdir(cur)
os.chdir('d4j_data')
base = os.getcwd()
list_project = os.listdir()
os.chdir(cur)

#project to be evaluated
list_project_title = ['Chart', 'Math', 'Time', 'Lang']
project_title = list_project_title[3]
list_project = [x for x in list_project if project_title in x]
deprecated_bugs = ['Math_6', 'Math_38', 'Lang_2', 'Time_21']
mutant_source = [x for x in list_project if x not in deprecated_bugs]
percentage = 0.9  # keep 10%
mutant_source = ['Time_27']

for p in mutant_source:
    mutants = {}
    method_mutants = {}
    major_log_path = f'./older_major/major_logs/{p}'
    with open(os.path.join(major_log_path, 'mutants.log'), newline='') as m_file:
        for line in m_file:
            mutant_info = line.strip().split(':')
            if len(mutant_info) != 7:
                index = len(mutant_info) - 1
                while not mutant_info[index].isdigit():
                    index -= 1
            else:
                index = 5
            mutant_no = int(mutant_info[0])
            method_name = mutant_info[index-1]
            if method_name not in method_mutants:
                method_mutants[method_name] = []
                method_mutants[method_name].append(mutant_no)
            else:
                method_mutants[method_name].append(mutant_no)
    sampled_mutant = []
    for method in method_mutants:
        list_mutant = method_mutants[method]
        sample_size = int(len(list_mutant) * percentage)
        sampled = random.sample(list_mutant, sample_size)
        sampled_mutant.extend(sampled)
    with open(f'exclude_mutant_sample/{p}_{1 - percentage:.1f}.log', 'w') as f:
        for mutant in sampled_mutant:
            f.write(f"{mutant}\n")
