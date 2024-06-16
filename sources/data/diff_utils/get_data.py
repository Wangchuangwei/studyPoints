import os
import json
import re
from multiprocessing import cpu_count, Pool
import difflib
import logging
import jpype.imports
from jpype.types import *

from transformers import RobertaTokenizer,RobertaConfig, RobertaModel

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'unixcoder':(RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['unixcoder']
tokenizer = tokenizer_class.from_pretrained("microsoft/unixcoder-base")

def read_json_file(filename):
    logger.info(f"当前的数据集是： {filename}")
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
        data = [json.loads(line) for line in data]
    return data

def save_json_data(data_dir, filename, data):
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, filename)
    with open(file_name, 'w') as output:
        if type(data) == list:
            if type(data[0]) in [str, list,dict]:
                for item in data:
                    output.write(json.dumps(item))
                    output.write('\n')

            else:
                json.dump(data, output)
        elif type(data) == dict:
            json.dump(data, output)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    print("saved dataset in " + file_name)

def compute_code_diffs(old_tokens, new_tokens):
    spans = []
    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(None, old_tokens, new_tokens).get_opcodes():
        if edit_type == 'equal':
            spans.extend("KEEP " + old_tokens[o_start:o_end]) 
        elif edit_type == 'replace':
            spans.extend("DEL " + old_tokens[o_start:o_end] + "ADD " + new_tokens[n_start:n_end])
        elif edit_type == 'insert':
            spans.extend("ADD " + new_tokens[n_start:n_end])
        else:
            spans.extend("DEL " + old_tokens[o_start:o_end])

    return spans

def get_contextual_medit(one_diff):
    old_tokens, new_tokens = one_diff["old"], one_diff["new"]
    diff = compute_code_diffs(old_tokens, new_tokens)
    result = {"diff":diff}
    return result

def get_tokenize_line_code(line_code):
    #token化
    text = " ".join(line_code)
    try:
        tokens_ori = tokenizer.tokenize(text)
    except: 
        tokens_ori = []
    if len(tokens_ori) == 0:
        return []
    return tokens_ori

def get_token_level(before, after):
    diff = difflib.SequenceMatcher(None, before, after)
    token_level_a = [0] * len(before)
    token_level_b = [0] * len(after)
    for op, a_i, a_j, b_i, b_j in diff.get_opcodes():
        a_tokens = before[a_i:a_j]
        b_tokens = after[b_i:b_j]
        if op == 'replace':
            for i in range(a_i, a_j):
                token_level_a[i] = 1
            for i in range(b_i, b_j):
                token_level_b[i] = 2
    token_level_a += token_level_b
    return token_level_a

def _heuristic_replace_match(a_tokens, b_tokens):
    diff_seqs = []
    code_diffs = []
    line_level = []
    token_level = []

    a_len = len(a_tokens)
    b_len = len(b_tokens)
    delta_len = max(a_len - b_len, b_len - a_len)
    if a_len != b_len:
        head_ratio = difflib.SequenceMatcher(None, a_tokens[0], b_tokens[0]).quick_ratio()
        tail_ratio = difflib.SequenceMatcher(None, a_tokens[-1], b_tokens[-1]).quick_ratio()
        if head_ratio >= tail_ratio:
            if a_len > b_len:
                b_tokens += [""] * delta_len
            else:
                a_tokens += [""] * delta_len
        else:
            if a_len > b_len:
                b_tokens = [""] * delta_len + b_tokens
            else:
                a_tokens = [""] * delta_len + a_tokens
    assert len(a_tokens) == len(b_tokens)
    for at, bt in zip(a_tokens, b_tokens):
        if at == "":
            diff_seqs.append([at, bt, "insert"])
            code_diffs.append('ADD ' + at)
            code_line = get_tokenize_line_code(['ADD ' + at]) 
            line_level += [2] * len(code_line)
            token_level += [0] * len(code_line)
        elif bt == "":
            diff_seqs.append([at, bt, "delete"])
            code_diffs.append('DEL ' + at)
            code_line = get_tokenize_line_code(['DEL ' + at])
            line_level += [1] * len(code_line)
            token_level += [0] * len(code_line)
        else:
            diff_seqs.append([at, bt, "replace"])
            code_diffs.append('DEL ' + at)
            code_line1 = get_tokenize_line_code(['DEL ' + at])
            line_level += [1] * len(code_line1)
            code_diffs.append('ADD ' + bt)
            code_line2 = get_tokenize_line_code(['ADD ' + bt])
            line_level += [2] * len(code_line2)
            token_level += get_token_level(code_line1, code_line2)
    return diff_seqs, code_diffs, line_level, token_level

def get_ast_diff():
    path = os.path.join(os.path.realpath("."), "jars/*")
    path1 = os.path.join(os.path.realpath("."), "before.java")
    path2 = os.path.join(os.path.realpath("."), "after.java")
    try:
        jpype.startJVM(classpath=[path])
    except Exception as e:
        pass

    MyClass = JClass('test')
    MyClass.main([])
    args = [JString("diff"),JString(path1),JString(path2)]
    diff_res = str(MyClass.parseDiff(args))

    ast_diff = diff_res.split('\n')
    actions = []

    for action in ast_diff:
        if not action.startswith('Match') and action:
            action_name = re.sub(u"\\(.*?\)", "", action.split(' at ')[0].replace(':', ''))
            simple_act = action_name.split(' ')[0] + action_name.split(' ')[1]
            actions.append(simple_act)
    return actions

def check_tokenize_line(lines_code):
    for line_code in lines_code:
        text = " ".join(line_code)
        try:
            tokens_ori = tokenizer.tokenize(text)
        except:
            return False
    return True

def get_code_ast_diff(diff_data):
    code_diffs = []
    ast_diffs = []
    docs = []
    code_befores = []
    code_afters = []

    lines_level = []
    tokens_level = []

    j = 0
    k = 0

    for idx, one_diff in enumerate(diff_data):
        k += 1
        print('-------->当前处理的数据是第：', k)

        #get diff code(eg. example) and msg from one_diff
        example = one_diff["diff"].split("<nl>")
        example = [item.strip() for item in example if len(item.strip()) > 0]
        msg = one_diff["msg"].lower().replace("\n", "").split()

        #get old and new version from diff code
        before_split_code = [example[1]]
        after_split_code = [example[0]]
        for item in example[2:]:
            if item[0] == "-":
                before_split_code.append(item[1:])
            elif item[0] == "+":
                after_split_code.append(item[1:])
            else:
                before_split_code.append(item)
                after_split_code.append(item)

        #vaild current example whether get ast diff and code diff
        if (not check_tokenize_line(before_split_code)) or (not check_tokenize_line(after_split_code)):
            continue 
        
        #write old and new version into corresponding .java and get ast diff information
        before_path = os.path.join(os.path.realpath("."), "before.java")
        after_path = os.path.join(os.path.realpath("."), "after.java")
        with open(before_path, 'w', encoding='utf-8') as f1:
            f1.write("public class Test {" + " ".join([line + '\n' for line in before_split_code]) + "}")
        with open(after_path, 'w', encoding='utf-8') as f1:
            f1.write("public class Test {" + " ".join([line + '\n' for line in after_split_code]) + "}")
        ast_diff = []
        ast_diff = get_ast_diff()

        #get code diff, line_level and token_level information for old and new version
        #eg. from before_split_code and after_split_code
        diff = difflib.SequenceMatcher(None, before_split_code, after_split_code)
        diff_seqs = []

        code_diff = []
        line_level = []
        token_level = []
        for op, a_i, a_j, b_i, b_j in diff.get_opcodes():
            a_tokens = before_split_code[a_i:a_j]
            b_tokens = after_split_code[b_i:b_j]
            if op == "delete":
                for at in a_tokens:
                    diff_seqs.append([at, "", op])
                    code_diff.append('DEL ' + at)
                    code_line = get_tokenize_line_code(['DEL ' + at])
                    line_level += [1] * len(code_line)
                    token_level += [0] * len(code_line)
            elif op == "insert":
                for bt in b_tokens:
                    diff_seqs.append(["", bt, op])
                    code_diff.append('ADD ' + bt)
                    code_line = get_tokenize_line_code(['ADD ' + bt])
                    line_level += [2] * len(code_line)
                    token_level += [0] * len(code_line)
            elif op == "equal":
                for at, bt in zip(a_tokens, b_tokens):
                    diff_seqs.append([at, bt, op])
                    code_diff.append('KEEP ' + at)
                    code_line = get_tokenize_line_code(['KEEP ' + at])
                    line_level += [0] * len(code_line)
                    token_level += [0] * len(code_line)
            else:
                # replace
                diff_seqs_a, code_diffs_a, line_level_a, token_level_a = _heuristic_replace_match(a_tokens, b_tokens)
                diff_seqs += diff_seqs_a
                code_diff += code_diffs_a
                line_level += line_level_a
                token_level += token_level_a


        print("--------split------")
        print("".join([line + '\n' for line in before_split_code]))
        print("".join([line + '\n' for line in after_split_code]))
        ans = []
        for item in code_diff:
            # ans.append(tokenizer.tokenize(item))
            ans += tokenizer.tokenize(item)
        print(len(ans), len(line_level), len(token_level))
        # print(len(line_level), len(token_level))
        print('-----test code diff-----')
        print('code_diff:', code_diff)
        print('-----test ast diff-----')
        print("ast_diff:", ast_diff)

        code_diffs.append(code_diff)
        ast_diffs.append(ast_diff)
        docs.append(msg)
        code_befores.append([" ".join(before_split_code)])
        code_afters.append([" ".join(after_split_code)])

        lines_level.append(line_level)
        tokens_level.append(token_level)
        j += 1
        print('-------->当前处理好的数据有：', j,'个')
        print('\n\n\n')
        # if j == 2:
        #     break

    return code_diffs, ast_diffs, docs, code_befores, code_afters, lines_level, tokens_level



if __name__ == '__main__':
    # for part in ["train", "vaild", "test"]:
        part = "my_test"
        filename = os.path.join(os.path.realpath('.'), 'java', "%s.jsonl"%part)
        diff_data = read_json_file(filename)

        #pre-processing data
        # examples = []
        # commit_msgs = []

        # for idx, one_diff in enumerate(diff_data ) :
        #     test_diff_ex = one_diff["diff"].split("<nl>")
        #     test_diff_ex =  [item.strip()   for item in test_diff_ex if len(item.strip()) > 0]
        #     examples.append(test_diff_ex)  
        #     commit_msgs .append(one_diff["msg"].lower().replace("\n", "").split())

        code_diffs, ast_diffs, docs, code_befores, code_afters, lines_level, tokens_level = get_code_ast_diff(diff_data)

        # data = []
        # for example in examples:
        #     data.append(get_old_new(example))
        # print(data[0])
        # cores = cpu_count()
        # pool = Pool(cores)
        # results = pool.map(get_old_new, examples)
        # pool.close()
        # pool.join()

        # cores = cpu_count()
        # pool = Pool(cores)
        # medit = pool.map(get_contextual_medit, results)
        # pool.close()
        # pool.join()

        # print(medit[0])
       