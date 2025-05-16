import subprocess
import torch
import os
from tqdm import tqdm
import json
import pickle
import re
from spiral import ronin
from rank_bm25 import BM25Okapi
import javalang

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

deprecated_bugs = ['Math_6', 'Math_38', 'Lang_2', 'Time_21']

# main process
# blocked code for identifier
batch = project_title
for mutant_source in list_mutant:
    batch_vocab = []
    batch = project_title
    average_dl = []
    for project_name in tqdm(mutant_source):
        if project_name in deprecated_bugs:
            continue
        os.chdir(f'd4j_data_fix/{project_name}')
        method = json.load(open('snippet.json'))
        test_method = json.load(open('test_snippet.json'))
        os.chdir(cur)
        project = project_name.split('_')[0]
        project_version = project_name.split('_')[1]
        LOG_PATH = "./major_logs/major_logs/"
        FT_PATH = "./failing_tests"
        FAULTY_COMPONENTS = get_faulty_methods(project, project_version)
        FAILING_TESTS = get_failing_tests(project, project_version, FT_PATH)
        COVERED_METHODS = get_covered_methods(project, project_version)
        batch+=project_version
        #mutant loading
        kill_reason = None
        project_log = LOG_PATH+project+'_'+project_version
        ALL_MUTANTS = parse_major_log(project_log, kill_reason)
        last_id = len(ALL_MUTANTS)
        count = 0
        field_key = []
        for mutant_no in ALL_MUTANTS:
            if '@' not in ALL_MUTANTS[mutant_no]['method_name']:
                field_key.append(mutant_no)
        for key in field_key:
            del ALL_MUTANTS[key]
        last_class = []
        method_dict = {}
        for m in method:
            ms = m['signature']
            cs = m['class_name']
            begin = m['begin_line']
            signature_key = ms.replace(' ', '')
            if 'Anonymous' in signature_key:
                an_identifier = signature_key.split('(')[0].split('.')[-2]
                last_class.append(signature_key.replace(an_identifier, ''))
                c_id = last_class.count(signature_key.replace(an_identifier, ''))
                signature_key = signature_key.replace(an_identifier, str(c_id))
            if cs not in ms:
                if '.' not in str(ms.split('(')[0]):
                    signature_key = cs+'.'+signature_key
            if '...' in signature_key:
                alt_signature = signature_key.replace('...', '[]')
                method_dict[(alt_signature, begin)] = m
            method_dict[(signature_key, begin)] = m

        last_line = []
        for mutant_no in ALL_MUTANTS:
            signature = ALL_MUTANTS[mutant_no]['method_name']
            signature = signature.replace(' ', '')
            signature = signature.replace('$', '.')
            m_line = ALL_MUTANTS[mutant_no]['line_no']
            if '<init>' in signature:
                class_name = signature.split('@')[0].split('.')[-1]
                signature = signature.replace('@<init>', '.'+class_name)
            else:
                signature = signature.replace('@', '.', 1)
            #make exception here
            sig_list = [key[0] for key in  method_dict.keys()]
            if signature not in sig_list:
                params = signature.split('(')[1].split(')')[0]
                if len(params) != 0:
                    list_p = params.split(',')
                    new_p = []
                    for p in list_p:
                        new_p.append(p.split('.')[-1])
                    signature = signature.split('(')[0]+'('+','.join(new_p)+')'
            if signature not in sig_list:
                continue
            signature_key = find_closest_key(signature, m_line, method_dict)
            if signature_key[1]==0:
                continue
            method_mutant = method_dict[signature_key]
            snippet = method_mutant['snippet']
            line_index = m_line - method_mutant['begin_line']
            line_strip = snippet.split('\n')
            before_op = ALL_MUTANTS[mutant_no]['before']
            after_op = ALL_MUTANTS[mutant_no]['after']
            if line_index<0 or line_index>=len(line_strip):
                continue
            mutated_line = line_strip[line_index]
            if '<NO-OP>' in after_op:
                after_op = ''
            if len(last_line) == 0 or ALL_MUTANTS[mutant_no]['line_no'] != last_line[0]:
                last_line = []
                last_line.append(ALL_MUTANTS[mutant_no]['line_no'])
                mutation = replace_nth_occurrence(mutated_line, before_op, after_op, 1)
                last_line.append((before_op, after_op))
            else:
                occ = last_line.count((before_op, after_op))
                mutation = replace_nth_occurrence(mutated_line, before_op, after_op, occ+1)
                last_line.append((before_op, after_op))
            line_strip[line_index] = mutation
            code_snip = '\n'.join(line_strip)
            tokens = spiral_token(code_snip)
            ALL_MUTANTS[mutant_no]['snippet'] = code_snip
            ALL_MUTANTS[mutant_no]['token'] = tokens
            ALL_MUTANTS[mutant_no]['signature'] = signature_key[0]
            batch_vocab.append(tokens)
        mutants = { mutant_no: mutant_info for mutant_no, mutant_info in ALL_MUTANTS.items()
                    if convert_signature(mutant_info['method_name']) in COVERED_METHODS }
        cm = {}
        for key in mutants:
            m = mutants[key]
            c = m['method_name']
            if c not in cm:
                cm[c] = []
            cm[c].append(m['killer'])
        ALL_TESTS = []
        for c in cm:
            m = cm[c]
            ALL_TESTS += sum(m, [])
        ALL_TESTS = list(set(ALL_TESTS))
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

        ALL_TESTS_CODING = {}
        flag = True
        for test in ALL_TESTS:
            for entry in truncated_test:
                testcode = entry[0]
                test_name = entry[1]
                test_name = [convert_signature(x) for x in test_name]
                if test in test_name[-1]:
                    ALL_TESTS_CODING[test] = (testcode, spiral_token(testcode))
                    batch_vocab.append(spiral_token(testcode))
                    break
                else:
                    test_name = [x.split('(')[0] for x in test_name]
                    if test in test_name:
                        ALL_TESTS_CODING[test] = (testcode, spiral_token(testcode))
                        batch_vocab.append(spiral_token(testcode))
                        break
        os.chdir(f'd4j_data_fix/{project_name}')
        
        mutant_embedding = {}
        for mutant_no in mutants:
            if 'snippet' not in mutants[mutant_no].keys():
                continue
            code_token = mutants[mutant_no]['token']
            tag = 'b' if mutants[mutant_no]['killer'] else 'nb'
            mutant_embedding[mutant_no] = {
                'method_name': mutants[mutant_no]['method_name'],
                'signature': mutants[mutant_no]['signature'],
                'killer': mutants[mutant_no]['killer'],
                'tag' : tag,
                'snippet': mutants[mutant_no]['snippet'],
                'token': mutants[mutant_no]['token']
            }
            average_dl.append(len(code_token))
        with open('mutant_data_bm25.pkl', 'wb') as mf:
            pickle.dump(mutant_embedding, mf)
        fix_method_embedding = {}
        for m in method:
            last_id += 1
            ms = m['signature']
            code = m['snippet']
            code_token = spiral_token(code)
            fix_method_embedding[str(last_id)] = {
                'method_name' : ms,
                'killer' : [],
                'tag': 'nc',
                'snippet': m['snippet'],
                'token': code_token
            }
            average_dl.append(len(code_token))
        with open('method_data_bm25.pkl', 'wb') as mef:
            pickle.dump(fix_method_embedding, mef)
        test_embedding = {}
        for test in ALL_TESTS_CODING:
            code = ALL_TESTS_CODING[test][0]
            code_token = ALL_TESTS_CODING[test][1]
            test_embedding[test] = {
                'snippet': code,
                'token': code_token
            }
            average_dl.append(len(code_token))
        with open('test_data_bm25.pkl', 'wb') as tf:
            pickle.dump(test_embedding, tf)
        os.chdir(cur)
    bm25 = BM25Okapi(batch_vocab)
    vocab = list(bm25.idf.keys())
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    os.makedirs('vocab', exist_ok=True)
    with open(f"vocab/{batch}_vocab.pkl", "wb") as f:
        pickle.dump({
        "bm25": bm25,
        "vocab_index": vocab_index,
        "adl": sum(average_dl)/len(average_dl)
    }, f)
    
        