import subprocess
import numpy as np
import os
from tqdm import tqdm
import json
import pickle
import re
from spiral import ronin
from rank_bm25 import BM25Okapi
import javalang
from collections import Counter

#stem base directory
cur = "c:/Users/COINSE/Downloads/simfl-extension"
os.chdir(cur)
os.chdir('d4j_data')
base = os.getcwd()
list_project = os.listdir()
os.chdir(cur)


#project to be evaluated
list_project_title = ['Chart', 'Math', 'Time', 'Lang']
project_title = list_project_title[2]
list_project = [x for x in list_project if project_title in x]
deprecated_bugs = ['Math_6', 'Math_38', 'Lang_2', 'Time_21']
list_project = [x for x in list_project if x not in deprecated_bugs]
list_project = sorted(list_project, key=lambda x: int(x.split('_')[1]), reverse=True)
model_version = project_title
list_mutant = [list_project[i:i+5] for i in range(0, len(list_project), 5)]

# utility function from simfl-source
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
def parse_major_log(major_log_path, kill_reason=None):
    if kill_reason == None:
        kill_reason = ['FAIL', 'TIME', 'EXC']

    test_map = {}
    with open(os.path.join(major_log_path, 'testMap.csv')) as tm_file:
        tm_file.readline()
        for line in tm_file.readlines():
            test_no, test_name = tuple(line.strip().split(','))
            test_map[int(test_no)] = test_name

    mutants = {}
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
            change = mutant_info[index+1:]
            if len(change)>1:
                change = [':'.join(change)]
            change = change[0]
            before = change.split('|==>')[0]
            after = change.split('|==>')[1]
            mutants[mutant_no] = {
                'class_name': method_name.split('@')[0],
                'method_name': method_name,
                'line_no': int(mutant_info[index]),
                'killer': [],
                'before':before,
                'after': after
            }

    with open(os.path.join(major_log_path, 'killMap.csv')) as km_file:
        km_file.readline()
        for line in km_file.readlines():
            test_no, mutant_no, reason = tuple(line.strip().split(',')[:3])
            if reason in kill_reason:
                test_name = test_map[int(test_no)].replace('[', '.').replace(']', '')
                if int(mutant_no) in mutants:
                    mutants[int(mutant_no)]['killer'].append(test_name)
    return mutants
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

def replace_nth_occurrence(text, old, new, nth):
    parts = text.split(old)
    
    if len(parts) > nth:
        return old.join(parts[:nth]) + new + old.join(parts[nth:])
    else:
        return text
def find_closest_key(signature, line, m_dict):
    keys = m_dict.keys()
    possible_key = [k[1] for k in keys if k[0] == signature]
    chosen_k = 0
    for k in possible_key:
        if k<=line:
            chosen_k = k
    return signature, chosen_k

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


def format_code(code):
    result = subprocess.run(['astyle'], input=code, capture_output=True, text=True)
    return result.stdout

def remove_java_comments(java_code):
    DEFAULT = 0
    IN_LINE_COMMENT = 1
    IN_BLOCK_COMMENT = 2
    IN_STRING = 3
    
    current_state = DEFAULT
    result = []
    i = 0
    
    while i < len(java_code):
        char = java_code[i]
        if current_state == DEFAULT:
            if i < len(java_code) - 1 and char == '/' and java_code[i + 1] == '/':
                current_state = IN_LINE_COMMENT
                i += 2
                continue
            elif i < len(java_code) - 1 and char == '/' and java_code[i + 1] == '*':
                current_state = IN_BLOCK_COMMENT
                i += 2
                continue
            elif char == '"' or char == "'":
                result.append(char)
                current_state = IN_STRING
                string_delimiter = char
                i += 1
                continue
            else:
                result.append(char)
                i += 1
                continue
                
        elif current_state == IN_LINE_COMMENT:
            if char == '\n':
                result.append('\n')  # Keep the newline
                current_state = DEFAULT
            i += 1
            continue
            
        elif current_state == IN_BLOCK_COMMENT:
            if i < len(java_code) - 1 and char == '*' and java_code[i + 1] == '/':
                current_state = DEFAULT
                i += 2
                continue
            else:
                i += 1
                continue
                
        elif current_state == IN_STRING:
            result.append(char)
            if char == '\\' and i + 1 < len(java_code):
                # Handle escape sequences in strings
                result.append(java_code[i + 1])
                i += 2
                continue
            elif char == string_delimiter:
                current_state = DEFAULT
            i += 1
            continue
    
    return ''.join(result)

def spiral_token(code):
    code = remove_java_comments(code)
    tk = list(javalang.tokenizer.tokenize(code))
    filtered = [t.value for t in tk if not isinstance(t, javalang.tokenizer.Keyword)]
    code = ' '.join(filtered)
    for w in ["\n", ";", "[", "]", "}", "{", "(", ")", ",", ".", "\"\""]:
        code = code.replace(w, " ")
    code = re.sub(r'"[^"]+"', " ", code)
    for w in ["\'", "\""]:
        code = code.replace(w, " ")
    code = re.sub(r'0[xX][0-9a-fA-F]+|\d+', " <num> ", code)
    code = re.sub(r"\b[A-Za-z0-9-_]{1}\b", " ", code)
    code = code.lower()
    code = ronin.split(code)
    return code

#def bm25_vector(tokenized_query, bm25, vocab_index):
#    vec = np.zeros(len(vocab_index), dtype=np.float32)
#    for token in tokenized_query:
#        if token in vocab_index:
#            idx = vocab_index[token]
#            vec[idx] = bm25.idf.get(token, 0.0)
#    return vec

def bm25_vector(tokenized_query, bm25, vocab_index, k1=1.5, b=0.75, avgdl=None):
    vec = np.zeros(len(vocab_index), dtype=np.float32)
    tf_counter = Counter(tokenized_query)
    doc_length = len(tokenized_query)

    for token, freq in tf_counter.items():
        if token in vocab_index:
            idx = vocab_index[token]
            idf = bm25.idf.get(token, 0.0)
            tf = freq
            # Apply BM25 formula
            score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avgdl))))
            vec[idx] = score
    return vec

deprecated_bugs = ['Math_6', 'Math_38', 'Lang_2', 'Time_21']

# main process
# blocked code for identifier
batch = project_title
for i in range(len(list_mutant)):
    mutant_source = list_mutant[i]
    inf_source = []
    if i+1<len(list_mutant):
        inf_source = list_mutant[i+1]
    batch = project_title
    for m in mutant_source:
        batch+=m.split('_')[1]
    with open(f'vocab/{batch}_vocab.pkl', 'rb') as vf:
        data = pickle.load(vf)
        bm25 = data["bm25"]
        vocab_index = data["vocab_index"]
        adl = data['adl']
    for project_name in tqdm(mutant_source):
        if project_name in deprecated_bugs:
            continue
        if project_title in project_name and project_name in mutant_source:
            project = project_name.split('_')[0]
            project_version = project_name.split('_')[1]
            os.chdir(f'd4j_data_fix/{project_name}')
            with open('mutant_data_bm25.pkl', 'rb') as mf:
                mutants = pickle.load(mf)
            with open('test_data_bm25.pkl', 'rb') as tf:
                test = pickle.load(tf)
            with open('method_data_bm25.pkl', 'rb') as mef:
                method = pickle.load(mef)
            mutant_embedding = {}
            for mutant_no in mutants:
                if 'snippet' not in mutants[mutant_no].keys():
                    continue
                code_token = mutants[mutant_no]['token']
                embedding = bm25_vector(code_token, bm25, vocab_index, avgdl=adl)
                tag = 'b' if mutants[mutant_no]['killer'] else 'nb'
                mutant_embedding[mutant_no] = {
                    'method_name': mutants[mutant_no]['method_name'],
                    'signature': mutants[mutant_no]['signature'],
                    'killer': mutants[mutant_no]['killer'],
                    'embedding': embedding,
                    'tag' : tag,
                    'snippet': mutants[mutant_no]['snippet'],
                    'token': mutants[mutant_no]['token']
                }
            with open('mutant_data_bm25.pkl', 'wb') as mf:
                pickle.dump(mutant_embedding, mf)
            fix_method_embedding = {}
            for method_no in method:
                code_token = method[method_no]['token']
                embedding = bm25_vector(code_token, bm25, vocab_index, avgdl=adl)
                fix_method_embedding[method_no] = {
                    'method_name' : method[method_no]['method_name'],
                    'killer' : [],
                    'embedding' : embedding,
                    'tag': method[method_no]['tag'],
                    'snippet': method[method_no]['snippet'],
                    'token': method[method_no]['token']
                }
            with open('method_data_bm25.pkl', 'wb') as mef:
                pickle.dump(fix_method_embedding, mef)
            test_embedding = {}
            for t in test:
                code_token = test[t]['token']
                embedding = bm25_vector(code_token, bm25, vocab_index, avgdl=adl)
                test_embedding[t] = {
                    'embedding': embedding,
                    'snippet': test[t]['snippet'],
                    'token': test[t]['token']
                }
            with open('test_data_bm25.pkl', 'wb') as tf:
                pickle.dump(test_embedding, tf)
            os.chdir(cur)

    for project_name in tqdm(inf_source):
        if project_name in deprecated_bugs:
            continue
        method_snip = json.load(open(f'd4j_data/{project_name}/snippet.json'))
        test_method = json.load(open(f'd4j_data/{project_name}/test_snippet.json'))
        project = project_name.split('_')[0]
        project_version = project_name.split('_')[1]
        LOG_PATH = "./major_logs/major_logs/"
        FT_PATH = "./failing_tests"
        FAULTY_COMPONENTS = get_faulty_methods(project, project_version)
        FAILING_TESTS = get_failing_tests(project, project_version, FT_PATH)
        COVERED_METHODS = get_covered_methods(project, project_version)
        os.chdir(f'd4j_data/{project_name}')
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
        COVERED_METHODS = [x for x in COVERED_METHODS if x not in UNCOVERED_METHODS]
        covered_method_embedding = {}
        for method_signature in COVERED_METHODS_CODING:
            code = COVERED_METHODS_CODING[method_signature]
            code_token = spiral_token(code)
            embedding = bm25_vector(code_token, bm25, vocab_index, avgdl=adl)
            covered_method_embedding[method_signature] = {
                'embedding': embedding
            }
        with open('buggy_method_data_bm25.pkl', 'wb') as bmef:
            pickle.dump(covered_method_embedding, bmef)
        failing_test_embedding = {}
        for test in FAILING_TESTS_CODING:
            code = FAILING_TESTS_CODING[test]
            code_token = spiral_token(code)
            embedding = bm25_vector(code_token, bm25, vocab_index, avgdl=adl)
            failing_test_embedding[test] = {
                'embedding': embedding
            }
        with open('failing_test_data_bm25.pkl', 'wb') as ftf:
            pickle.dump(failing_test_embedding, ftf)
        os.chdir(cur)
    
    
        