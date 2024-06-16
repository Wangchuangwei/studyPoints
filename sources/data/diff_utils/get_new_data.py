## 用于处理研究点二，带有标签的数据集

import jpype.imports
from jpype.types import *
import os
import logging
import javalang
import re
import difflib
import pickle

import torch
from transformers import RobertaTokenizer,RobertaConfig, RobertaModel

from .differ import fileLineDiff
from .diff.myers import myers_diff
from .diff.common import Keep, Insert, Remove

from .get_ast_root_action import get_ast_root

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'unixcoder':(RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['unixcoder']
tokenizer = tokenizer_class.from_pretrained("./unixcoder-base")
# k = 0

def _heuristic_replace_match(a_tokens, b_tokens):
    # global k

    diff_seqs = []
    #new add
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
            # if k == 4:
            #     print('---insert2----')
            #     # print('ADD '+ at)
            #     print(len(tokenizer.tokenize('ADD '+ at)), len(code_line))
            
            line_level += [2] * len(code_line)
            token_level += [0] * len(code_line)
        elif bt == "":
            diff_seqs.append([at, bt, "delete"])
            code_diffs.append('DEL ' + at)
            code_line = get_tokenize_line_code(['DEL ' + at])
            # if k == 4:
            #     print('---delete2----')
            #     print(len(tokenizer.tokenize('DEL '+ at)), len(code_line))

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

            # if k == 4:
            #     print('---replace2----')
            #     print(len(tokenizer.tokenize('DEL '+ at)), len(code_line1))
            #     print(len(tokenizer.tokenize('ADD '+ bt)), len(code_line2))

    return diff_seqs, code_diffs, line_level, token_level

def get_token_level(before, after):
    diff = difflib.SequenceMatcher(None, before, after)
    token_level_a = [0] * len(before)
    token_level_b = [0] * len(after)
    # print('---------------token_level-----------------')
    # print('befoer:',before)
    # print('after:', after)
    # print(token_level_a, token_level_b)
    # print('---------------token_level_action-----------------')
    # print(diff.get_opcodes())
    # token_level_a[1:3] = 1
    # print(token_level_a)
    for op, a_i, a_j, b_i, b_j in diff.get_opcodes():
        a_tokens = before[a_i:a_j]
        b_tokens = after[b_i:b_j]
        # print(a_i, a_j, b_i, b_j)
        # print("target_op:",op, a_tokens, b_tokens)
        if op == 'replace':
            for i in range(a_i, a_j):
                token_level_a[i] = 1
            for i in range(b_i, b_j):
                token_level_b[i] = 2
    # print(token_level_a, token_level_b)
    token_level_a += token_level_b
    # print(token_level_a)
    return token_level_a



def get_code_token(code_diff, code_tokens):
    # if j == 32 :
    #     logger.info(f"code_diff:{code_diff}")
    test1 = javalang.tokenizer.tokenize(code_diff)
    for i in list(test1):
        code_tokens.append(i.value)
    return code_tokens

def get_tokenize_line_code(line_code):
    #token化
    # config_class, model_class, tokenizer_class = MODEL_CLASSES['unixcoder']
    # tokenizer = tokenizer_class.from_pretrained("microsoft/unixcoder-base")
    text = " ".join(line_code)
    # print('text:', text)
    try:
        # tokens_ori = list(javalang.tokenizer.tokenize(text))
        tokens_ori = tokenizer.tokenize(text)
    except: 
        tokens_ori = []
    # print('tokens_ori is:', tokens_ori)
    if len(tokens_ori) == 0:
        return []
    # codes_ori = [x.value for x in tokens_ori]
    # return codes_ori
    # print('split tokenize:', len(tokens_ori))
    return tokens_ori

def check_tokenize_line(lines_code):
    # print(lines_code)
    for line_code in lines_code:
        text = " ".join(line_code)
        try:
            tokens_ori = list(javalang.tokenizer.tokenize(text))
        except:
            # print('the error split is :', text)
            return False
    return True


def get_code_ast_diff(file):
    code_diffs = []
    ast_diffs = []
    docs = []
    code_befores = []
    code_afters = []
    all_code_tokens = []
    diff_lines = []

    labels = []

    lines_level = []
    tokens_level = []

    new_code_diffs = []


    all_deletes = []
    all_updates = [] 
    # all_moves = []
    # all_adds = []
    all_old_nodes = []
    all_new_nodes = []

    # global k

    with open(file, 'rb') as f:
        j = 0
        k = 0
        obj = pickle.load(f)
        length = len(obj[0])
        for i in range(int(length * 0.9), length):
        # for i in range(int(length * 0.8)):
        # for i in range(int(length * 0.8), int(length * 0.9)):
            k += 1
            print('-------->当前处理的数据是第：', k)
            if obj[4][i] == '':
                continue
            comment = obj[4][i]
            label = obj[2][i]

            before_split_code = []
            after_split_code = []
            # before_split_code = obj[0][i]
            # after_split_code = obj[1][i]

            before_code_line = ""
            after = ""
            for item in obj[0][i]:
                before_code_line += item.strip()
                before_split_code.append(item.strip())
            for item in obj[1][i]:
                after += item.strip() 
                after_split_code.append(item.strip())

            result = fileLineDiff(before_split_code,after_split_code)

            # print('\n\n\n')
            if (not check_tokenize_line(before_split_code)) or (not check_tokenize_line(after_split_code)):
                # print("error split")
                continue 

            code_diff = []
            code_tokens = []
            diff_line = []

            try:
                for elem in result:
                    # print('elem:', elem[0].strip())
                    if isinstance(elem, Keep ):
                        code_tokens.append('KEEP '+ elem[0].strip() + '\n')
                        diff_line.append(0)
                    elif isinstance(elem, Insert):
                        code_tokens.append('ADD ' + elem[0].strip() + '\n')
                        diff_line.append(1)
                    elif isinstance(elem, Remove):
                        code_tokens.append('DEL ' + elem[0].strip() + '\n')
                        diff_line.append(2)
            except Exception as e:
                continue

            code_diff = ' '.join(code_tokens)

            # print('code diff is: ---\n', code_diff)
            # print('before_code_line is: ---\n', before_code_line)
            # print('after is:---\n', after)

            before_path = os.path.join(os.path.realpath("."), "data/diff_utils/before.java")
            after_path = os.path.join(os.path.realpath("."), "data/diff_utils/after.java")
            with open(before_path, 'w', encoding='utf-8') as f1:
                f1.write("public class Test {" + before_code_line + "}")
            with open(after_path, 'w', encoding='utf-8') as f1:
                f1.write("public class Test {" + after + "}")


            # get ast root
            print('-----start get ast root------')
            tem_path = '../gumtree/gumtree/bin' 
            root_old, old_nodes = get_ast_root(tem_path, 'before')
            root_new, new_nodes = get_ast_root(tem_path, 'after')
            all_delete, all_update = get_ast_action('before', 'after', root_old, root_new, tem_path)

            all_old_nodes.append(old_nodes)
            all_new_nodes.append(new_nodes)
            all_deletes.append(all_delete)   # for old ast change
            all_updates.append(all_update)   # for new ast change
            
            for node in old_nodes:
                print(node.pos, node.typeLabel, node.label)
            print('----res after get action------','\n')
            for j in range(len(all_delete)):
                cur_del = all_delete[j]
                print(cur_del.pos, cur_del.typ)
            print('-----end delete------')
            for j in range(len(all_update)):
                cur_del = all_update[j]
                print(cur_del.pos, cur_del.typ)
            print('-----end update------')


            ast_diff = []
            ast_diff = get_ast_diff()
            ast_diffs.append(ast_diff)
            # with open(os.path.join(os.path.realpath("."), "data/diff_utils/ast_diff.txt"), 'a', encoding='utf-8') as f:
            #     f.write(f"{file}")
            #     f.write(f"------->{j}")
            #     f.write(f"{get_ast_diff()}\n")

            # print('\n\n\n')
            # print("--------split------")
            # print("".join([line + '\n' for line in before_split_code]))
            # print("".join([line + '\n' for line in after_split_code]))
            # print('-------asd----')
            diff = difflib.SequenceMatcher(None, before_split_code, after_split_code)
            diff_seqs = []
            new_code_diff = []
            new_tokenize_code = []
            line_level = []
            token_level = []
            # print(diff.get_opcodes())
            for op, a_i, a_j, b_i, b_j in diff.get_opcodes():
                a_tokens = before_split_code[a_i:a_j]
                b_tokens = after_split_code[b_i:b_j]
                # print('aaaaa:',op, a_tokens, b_tokens)
                if op == "delete":
                    for at in a_tokens:
                        diff_seqs.append([at, "", op])
                        new_code_diff.append('DEL ' + at)
                        code_line = get_tokenize_line_code(['DEL ' + at])
                        # if k == 4:
                        #     print('---delete1----')
                        #     print(len(tokenizer.tokenize('DEL '+ at)), len(code_line))

                        line_level += [1] * len(code_line)
                        token_level += [0] * len(code_line)
                elif op == "insert":
                    for bt in b_tokens:
                        diff_seqs.append(["", bt, op])
                        new_code_diff.append('ADD ' + bt)
                        code_line = get_tokenize_line_code(['ADD ' + bt])
                        # if k == 4:
                        #     print('---insert1----')
                        #     print(len(tokenizer.tokenize('ADD '+ at)), len(code_line))

                        line_level += [2] * len(code_line)
                        token_level += [0] * len(code_line)
                elif op == "equal":
                    for at, bt in zip(a_tokens, b_tokens):
                        diff_seqs.append([at, bt, op])
                        # print(at, bt, op, 'haaaaaaaaa')
                        new_code_diff.append('KEEP ' + at)
                        # print('new code diff is:',new_code_diff)
                        code_line = get_tokenize_line_code(['KEEP ' + at])
                        # if k == 4:
                        #     print('---equal1----')
                        #     print(len(tokenizer.tokenize('KEEP '+ at)), len(code_line))
                        # print('equal is----------:', code_line)

                        line_level += [0] * len(code_line)
                        token_level += [0] * len(code_line)
                else:
                    # replace
                    # diff_seqs += _heuristic_replace_match(a_tokens, b_tokens)
                    diff_seqs_a, code_diffs_a, line_level_a, token_level_a = _heuristic_replace_match(a_tokens, b_tokens)
                    diff_seqs += diff_seqs_a
                    new_code_diff += code_diffs_a
                    line_level += line_level_a
                    token_level += token_level_a

            # print('diff_seqs',diff_seqs)
            # print('---------------------------------------------')
            # print('ast_diffs is:', ast_diff)
            # print('code_diffs:',new_code_diff)
            # print('----line_level')
            # print(line_level)
            # print("----token_level----")
            # print(token_level)

            # print('-----test length-----')
            # print('new_code_diff:', new_code_diff)
            # ans = []
            # for item in new_code_diff:
            #     # ans.append(tokenizer.tokenize(item))
            #     ans += tokenizer.tokenize(item)
            # # print(len(ans), len(line_level), len(token_level))
            # print('当前的数据是第',k,'个：', len(ans), len(line_level), len(token_level))
                
            # assert len(ans)==len(line_level)==len(token_level)
            

            code_diffs.append(code_diff)
            docs.append(comment)
            code_befores.append(before_code_line)
            code_afters.append(after)
            all_code_tokens.append(code_tokens)
            diff_lines.append(diff_line)

            labels.append(label)

            new_code_diffs.append(new_code_diff)
            lines_level.append(line_level)
            tokens_level.append(token_level)
            j += 1
            # print('-------->当前处理好的数据有：', j,'个')
            # with open(os.path.join(os.path.realpath("."), "data/diff_utils/middle_datasets/filter_datasets/test.tsv"), 'a', encoding='utf-8') as f1:
            #     f1.write(f"{old_line}")
                # f1.write("\n")
            # print('\n\n\n')
            if j == 10:
                break
        logger.info(f"当前的数据集是： {file}")
        logger.info(f"当前处理的数据共：{j}")
        # print(code_tokens)
        # print(code_diffs)
        # print(diff_lines)
        # print('------sdfsffff----')
        # print(code_diffs[0])
        # print(diff_lines[0])
        # print('------sdfsffff----')
        # print(code_diffs[1])
        # print(diff_lines[1])   


    # return code_diffs, ast_diffs, docs, code_befores, code_afters, all_code_tokens
    return new_code_diffs, ast_diffs, docs, labels,code_befores, code_afters, all_code_tokens, lines_level, tokens_level

# def read_ast_dataset

def get_ast_diff():
    path = os.path.join(os.path.realpath("."), "data/diff_utils/jars/*")
    # path = r'C:\Users\wang\Desktop\gumtree-3.0.0\lib'
    path1 = os.path.join(os.path.realpath("."), "data/diff_utils/before.java")
    path2 = os.path.join(os.path.realpath("."), "data/diff_utils/after.java")
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

def get_changed_idx(strr):
    if ':' in strr:
        typ, name_idx = strr.split(':')
        typ = typ.strip()
        name_idx = name_idx.strip()
        name = name_idx[:name_idx.index('(')]
        idx = name_idx[name_idx.index('('):]
        idx = idx.lstrip('(').rstrip(')')
        # return ActionNode(typ, int(idx), name)
        return int(idx)

    else:    
        typ = strr[:strr.index('(')]
        idx = strr[strr.index('('):]
        idx = idx.lstrip('(').rstrip(')')
        if typ == 'NullLiteral':
            # return ActionNode(typ, int(idx), 'null')
            return int(idx)
        if typ == 'ThisExpression':
            # return ActionNode(typ, int(idx), 'this')
            return int(idx)
        # return ActionNode(typ, int(idx))
        return int(idx)



def print_dirlist():
    # a = os.getcwd()
    # print(os.path.realpath(a))
    print(os.path.realpath("."))
    path = os.path.join(os.path.realpath("."), "data/diff_utils/accumulo/accumulo.pkl")
    # print(os.listdir(path))

    # path1 = os.path.join(os.path.realpath("."), "data/diff_utils/before.java")
    # path2 = os.path.join(os.path.realpath("."), "data/diff_utils/after.java")
    logger.info('jjjj')
    code_diffs, ast_diffs, docs, code_befores, code_afters, all_code_tokens, lines_level, tokens_level = get_code_ast_diff(path)
    print('\n\n\n')
    print('-----end-----')
    # print("code_diffs:\n",code_diffs)
    # print("ast_diffs:\n", ast_diffs)
    # # print("ast_diffs_1:\n", ast_diffs[0])
    # print("lines_level:", lines_level)
    # print("tokens_level:", tokens_level)
    pass

def filter_pointer():
    # a = os.getcwd()
    # print(os.path.realpath(a))
    print(os.path.realpath("."))
    path = os.path.join(os.path.realpath("."), "data/diff_utils/large_datasets/train.tsv")
    # print(os.listdir(path))

    # path1 = os.path.join(os.path.realpath("."), "data/diff_utils/before.java")
    # path2 = os.path.join(os.path.realpath("."), "data/diff_utils/after.java")
    logger.info('jjjj')
    code_diffs, ast_diffs, docs, code_befores, code_afters, all_code_tokens, lines_level, tokens_level = get_code_ast_diff(path)
    print('\n\n\n')
    print('-----end-----')
    # print("code_diffs:\n",code_diffs)
    # print("ast_diffs:\n", ast_diffs)
    # # print("ast_diffs_1:\n", ast_diffs[0])
    # print("lines_level:", lines_level)
    # print("tokens_level:", tokens_level)
    pass


if __name__ == '__main__':
    print_dirlist()



