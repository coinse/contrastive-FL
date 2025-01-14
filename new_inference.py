import os
import torch
import os
import json
import operator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pickle

#stem base directory
cur = "c:/Users/COINSE/Downloads/simfl-extension"
os.chdir(cur)
os.chdir('d4j_data')
base = os.getcwd()
list_project = os.listdir()
os.chdir(cur)

#project to be evaluated
list_project_title = ['Chart', 'Math', 'Time', 'Lang']
project_title = list_project_title[0]
list_project = [x for x in list_project if project_title in x]
model_version = project_title

#config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#utility function from simfl-source
def get_failing_tests(project, fault_no, ftc_path):
    file_path = os.path.join(ftc_path, project, str(fault_no))
    ftc = []
    with open(file_path, "r") as ft_file:
        for test_case in ft_file:
            ftc.append(test_case.strip())
    return ftc
def get_faulty_methods(project, fault_no):
    with open("buggy_methods/buggy_methods_{}_{}_list.csv".format(project, fault_no),
              'r') as bm_file:
        methods = [l.strip() for l in bm_file]
        return methods
def get_covered_methods(project, fault_no):
    with open("ft_method_covers_signature/{}/{}/covered.list".format(project, fault_no), 'r') as cm_file:
        methods = [l.strip() for l in cm_file]
        return methods
def convert_signature(flat_name):
    try:
        method = flat_name.split('(')[0]
        signature = flat_name.split('(')[1][:-1]
    except IndexError:
        return flat_name

    opened = 0
    replaced = ""
    for c in signature:
        if c == '<':
            opened += 1
        elif c == '>':
            opened -= 1
        else:
            if opened > 0:
                continue
            else:
                replaced += c
    return method + '(' + replaced + ')'
def method_to_signature(m_name, m_dict):
    signature = m_name
    signature = signature.replace(' ', '')
    signature = signature.replace('$', '.')
    if '<init>' in signature:
        class_name = signature.split('@')[0].split('.')[-1]
        signature = signature.replace('@<init>', '.'+class_name)
    else:
        signature = signature.replace('@', '.', 1)
    if signature not in m_dict.keys():
        params = signature.split('(')[1].split(')')[0]
        if len(params) != 0:
            list_p = params.split(',')
            new_p = []
            for p in list_p:
                new_p.append(p.split('.')[-1])
            signature = signature.split('(')[0]+'('+','.join(new_p)+')'
    return signature

#load model
s_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", trust_remote_code = True)
s_model.max_seq_length = 8192
s_model.eval()

from new_modelloss import ContrastiveModel
#main evaluation
#blocked code for when using all relevant test

text_list = []
arc = 'relu'
mod = 'euclidean'
ds_mod = ['zero']
strat = ['min', 'avg']
for ds in ds_mod:
    for strategy in strat:
        acc_stat = [0, 0, 0, 0]
        model_dict = torch.load(f'new-model/{model_version}/model_{arc}_{ds}.pth')
        model = ContrastiveModel(embedding_dim=768, projection_dim=768, output_dim=768, mode=mod)
        model.load_state_dict(model_dict)
        model.eval()
        uncovered_collage = []
        for p in tqdm(list_project):
            if project_title in p and p != 'Math_38' and p != 'Math_6':
                method_snip = json.load(open(f'd4j_data/{p}/snippet.json'))
                test_method = json.load(open(f'd4j_data/{p}/test_snippet.json'))
                cur = os.getcwd()
                #os.chdir(f'd4j_data_fix/{p}')
                #with open('test_data.pkl', 'rb') as tf:
                #    test_embedding = pickle.load(tf)
                #os.chdir(cur)
                project = p.split('_')[0]
                project_version = p.split('_')[1]
                LOG_PATH = "./major_logs/major_logs/"
                FT_PATH = "./failing_tests"
                FAULTY_COMPONENTS = get_faulty_methods(project, project_version)
                FAILING_TESTS = get_failing_tests(project, project_version, FT_PATH)
                COVERED_METHODS = get_covered_methods(project, project_version)
                acc1 = 0
                acc3 = 0
                acc5 = 0
                acc10 = 0
                truncated_test = []
                for entry in test_method:
                    testcode = entry['snippet']
                    labels = []
                    if len(entry['child_classes'])>0:
                        for child in entry['child_classes']:
                            alt_test_name = child+ '.'+entry['signature'].split('.')[-1]
                            labels.append(alt_test_name)
                    labels.append(entry['signature'])
                    truncated_test.append((testcode, labels))
                FAILING_TESTS_CODING = {}
                for test in FAILING_TESTS:
                    for entry in truncated_test:
                        testcode = entry[0]
                        test_name = entry[1]
                        test_name = [convert_signature(x) for x in test_name]
                        if test in test_name[-1]:
                            FAILING_TESTS_CODING[test] = testcode
                            break
                        else:
                            test_name = [x.split('(')[0] for x in test_name]
                            if test in test_name:
                                FAILING_TESTS_CODING[test] = testcode
                                break
                #ALL_TESTS_CODING = {}
                #for test in test_embedding:
                #    for entry in truncated_test:
                #        testcode = entry[0]
                #        test_name = entry[1]
                #        test_name = [convert_signature(x) for x in test_name]
                #        if test in test_name[-1]:
                #            ALL_TESTS_CODING[test] = testcode
                #            break
                #        else:
                #            test_name = [x.split('(')[0] for x in test_name]
                #            if test in test_name:
                #                ALL_TESTS_CODING[test] = testcode
                #                break
                last_class = []
                method_dict = {}
                for m in method_snip:
                    ms = m['signature']
                    cs = m['class_name']
                    signature_key = ms.replace(' ', '')
                    if 'Anonymous' in signature_key:
                        an_identifier = signature_key.split('(')[0].split('.')[-2]
                        last_class.append(signature_key.replace(an_identifier, ''))
                        c_id = last_class.count(signature_key.replace(an_identifier, ''))
                        signature_key = signature_key.replace(an_identifier, str(c_id))
                    if cs not in ms:
                        if '.' not in str(ms.split('(')[0]):
                            signature_key = cs+'.'+signature_key
                    method_dict[signature_key] = m
                COVERED_METHODS_CODING = {}
                UNCOVERED_METHODS = []
                SCORE = {}
                for method in COVERED_METHODS:
                    signature = method_to_signature(method, method_dict)
                    if signature in method_dict:
                        method_ = method_dict[signature]
                        snippet = method_['snippet']
                        COVERED_METHODS_CODING[signature] = snippet
                        SCORE[signature] = []
                    else:
                        UNCOVERED_METHODS.append(method)
                uncovered_collage.extend(UNCOVERED_METHODS)
                COVERED_METHODS = [x for x in COVERED_METHODS if x not in UNCOVERED_METHODS]
                for method_signature in COVERED_METHODS_CODING:
                    batch = []
                    method_snippet = [COVERED_METHODS_CODING[method_signature]]
                    method_output = s_model.encode(method_snippet)[0]
                    method_output = torch.from_numpy(method_output)
                    for test in FAILING_TESTS_CODING:
                        test_snippet = [FAILING_TESTS_CODING[test]]
                        test_output = s_model.encode(test_snippet)[0]
                        test_output = torch.from_numpy(test_output)
                        batch.append((test_output, method_output))
                    test_batch = torch.stack([x[0] for x in batch])
                    method_batch = torch.stack([x[1] for x in batch])
                    output = model(test_batch, method_batch)
                    SCORE[method_signature] = output.tolist()
                #for method_signature in COVERED_METHODS_CODING:
                #    batch = []
                #    method_snippet = [COVERED_METHODS_CODING[method_signature]]
                #    method_output = s_model.encode(method_snippet)[0]
                #    method_output = torch.from_numpy(method_output)
                #    for test in ALL_TESTS_CODING:
                #        test_snippet = [ALL_TESTS_CODING[test]]
                #        test_output = s_model.encode(test_snippet)[0]
                #        test_output = torch.from_numpy(test_output)
                #        batch.append((test_output, method_output))
                #    test_batch = torch.stack([x[0] for x in batch])
                #    method_batch = torch.stack([x[1] for x in batch])
                #    output = model(test_batch, method_batch)
                #    SCORE[method_signature] = output.tolist()
                scores = {}
                #multiplier = [1 if test in FAILING_TESTS else -1 for test in ALL_TESTS_CODING]
                for method in COVERED_METHODS:
                    signature = method_to_signature(method, method_dict)
                    #scores[method] = sum([x*y for x, y in zip(multiplier, SCORE[signature])])
                    if strategy == 'min':
                        scores[method] = min(SCORE[signature])
                    if strategy == 'avg':
                        scores[method] = sum(SCORE[signature])/len(SCORE[signature])
                    
                sorted_scores = list(sorted(scores.items(), key=operator.itemgetter(1)))
                def top_sorted(l, n, fc):
                    top_crop = l[:n]
                    yield_method = [x[0] for x in top_crop]
                    found = False
                    for method in yield_method:
                        if method in fc:
                            found = True
                            break
                    return found
                if sorted_scores[0][0] in FAULTY_COMPONENTS:
                    acc1+=1
                if acc1 or top_sorted(sorted_scores, 3, FAULTY_COMPONENTS):
                    acc3+=1
                if acc1 or acc3 or top_sorted(sorted_scores, 5, FAULTY_COMPONENTS):
                    acc5+=1
                if acc1 or acc3 or acc5 or top_sorted(sorted_scores, 10, FAULTY_COMPONENTS):
                    acc10+=1
                acc = [acc1, acc3, acc5, acc10]
                acc_stat = [x+y for x, y in zip(acc_stat, acc)]
        text_list.append('accstat for {}:'.format(f'new-model/{model_version}/model_{arc}_{mod}_{ds} with {strategy}'))
        text_list.append('acc@1: {}'.format(acc_stat[0]))
        text_list.append('acc@3: {}'.format(acc_stat[1]))
        text_list.append('acc@5: {}'.format(acc_stat[2]))
        text_list.append('acc@10: {}'.format(acc_stat[3]))
        text_list.append('\n')
uncovered_collage = list(set(uncovered_collage))
uncovered_constructor = [x for x in uncovered_collage if 'clinit' in x or 'class' in x or 'init' in x]
print(f'number of uncovered method across {model_version} : {len(uncovered_collage)}, constructor : {len(uncovered_constructor)}')
os.makedirs(f'results/{project_title}', exist_ok=True)
with open(f'results/{project_title}/new_{arc}_results.txt', 'w') as file:
    for line in text_list:
        file.write(line+'\n')




