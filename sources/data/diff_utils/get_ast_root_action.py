from distutils.log import debug
import json
import pickle
import os
import subprocess
from copy import deepcopy
all_keys = ['id', 'type', 'typeLabel', 'pos', 'length', 'children','label' ] 

# import jpype.imports
# from jpype.types import *
from transformers import RobertaTokenizer,RobertaConfig, RobertaModel

MODEL_CLASSES = {'unixcoder':(RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['unixcoder']
tokenizer = tokenizer_class.from_pretrained("microsoft/unixcoder-base")

class Node:
    def __init__(self):
        self.ori_id = None
        self.idx = None
        self.type = None
        self.label = None
        self.typeLabel = None
        self.pos = None
        self.length = None
        self.children = []
        self.father = None
    def __str__(self):
        return 'ori_id:%s idx:%s type:%s label:%s typeLabel:%s pos:%s length:%s'%(self.ori_id, self.idx, self.type, self.label, self.typeLabel, self.pos, self.length)
    def print_tree(self, depth):
        strr = '\t' * depth + str(self) + '\n'
        for child in self.children:
            strr += child.print_tree(depth + 1)
        return strr
    def get_all_nodes(self):
        nodes = []
        nodes.append(self)
        for node in self.children:
            nodes += node.get_all_nodes()
        return nodes

class ActionNode:
    def __init__(self, typ, idx, name=None):
        self.typ = typ
        self.idx = idx
        self.name = name
    
    def __eq__(self, node):
        return self.typ == node.typ and self.idx == node.idx and self.name == node.name

class ActionNode1:
    def __init__(self, typ, pos, label=None):
        self.typ = typ
        self.pos = pos
        self.label = label

    def __str__(self):
        return 'type:%s pos:%s label:%s'%(self.typ, self.pos, self.label)
    
    def __eq__(self, node):
        return self.typ == node.typ and self.pos == node.pos and self.label == node.label

def process_ast(ast):
    nodes = []
    
    node = Node()
    if 'label' in ast:
        node.label = ast['label']
    else:
        node.label = None
    
    # node.ori_id = int(ast['id'])
    # node.type = ast['type']
    node.typeLabel = ast['typeLabel']
    # node.pos = int(ast['pos'])
    node.pos = ast['pos']
    # node.length = ast['length']

    # print(type(ast), type(ast['children']))
    # print(node.label, node.type, node.typeLabel, node.pos, node.length, node.children)
    # print('a')

    if node.typeLabel == 'NullLiteral':
        assert node.label == None
        node.label = 'null'
    if node.typeLabel == 'ThisExpression':
        assert node.label == None
        node.label = 'this'
    nodes.append(node)

    for child in ast['children']:
        nodes += process_ast(child)
        nodes.append('^')
    return nodes

def parse_indented_string(indented_string):
    lines = indented_string.strip().split('\n')
    root = {"label": "root", 'typeLabel': None,'pos': None,"children": []}
    stack = []
    stack.append((0,root))
    prev_indent = -1

    i = 0
    ac = ""
    try: 
        for line in lines:
            ac = line
            line = line.rstrip()
            indent = len(line) - len(line.lstrip())
            if (':' in line.lstrip()):
                try:
                    typeLabel, type2 = line.lstrip().split(':',1)
                    # if i == 62:
                    #     print(line.lstrip())
                    #     print('haha')
                    #     print(typeLabel, type2, len(type2))
                    label, pos = type2.strip().rsplit()
                    # print('hhhh:',label, pos)
                except Exception as e:
                    print('120:',e)
                node = { "label": label.strip(), "typeLabel": typeLabel.strip(),"pos": pos.strip(),"children":[]}
            else:
                typeLabel, pos = line.lstrip().split()
                node = {"typeLabel":  typeLabel.strip(), "pos": pos.strip(),"children":[]}

            if indent > prev_indent:
                stack[-1][1].setdefault("children", []).append(node)
                stack.append((indent,node))
            else:
                while indent <= prev_indent:
                    stack.pop()
                    prev_indent = stack[-1][0]
                stack[-1][1].setdefault("children", []).append(node)
                stack.append((indent,node))
            prev_indent = indent
            i += 1
            # if i == 15:
            #     break
    except Exception as e:
        print('140:',e)
    return root


def get_ast_root(tem_path, file_name):
    tem_path1 = '../gumtree/gumtree'
    # file_name1 = 'parse_ast'

    tem_path = './data/diff_utils'   # C:\Users\wang\Desktop\gumtree-3.0.0\bin              2.1.2    3.0.0
    # file_name = 'Test2'
    out = subprocess.Popen('./data/diff_utils/gumtree/gumtree-3.0.0/bin/gumtree parse %s/%s.java'%(tem_path, file_name), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    # out = subprocess.Popen('C:/Users/wang/Desktop/gumtree-3.0.0/bin/gumtree parse %s/%s.java'%(tem_path, file_name), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout,_ = out.communicate()

    try:
        indented_string = stdout.decode('utf-8')
        ast = parse_indented_string(indented_string)
        # print(type(ast1))
        # ast2 = json.dumps(ast1)
        # print(ast1)
        # print(type(ast2), ast1['label'])
        # print(ast1.label)
        # print('label' in ast)
        # print(int('[12,15]'))
        # print('ha')
    except Exception as e:
        # print('b')
        print("166:",e)


    # try:
    #     ast = json.loads(stdout.decode('utf-8'))
    # except Exception as e:
    #     print(e)
    #     return None

    json.dump(ast, open('%s/%s.ast'%(tem_path1, file_name), 'w'), indent=1) 

    # print('----161---', ast)

    root = Node()
    root.label = 'root'
    root.pos = -1
    all_nodes = []
    all_nodes.append(root)
    # all_nodes += process_ast(ast['root'])
    all_nodes += process_ast(ast)
    all_nodes += ['^']

    # print('---------------------')
    # print(all_nodes)

    all_nodes_new = []
    root = all_nodes[0]
    root.idx = 0
    all_nodes_new.append(root)
    cur_node = root
    idx = 1
    for node in all_nodes[1:]:
        if node == '^':
            cur_node = cur_node.father
            # print('len:', cur_node.type,len(cur_node.children), 'idx:', cur_node.idx, 'label:', cur_node.label)
        else:
            node.idx = idx
            node.father = cur_node
            all_nodes_new.append(node)
            cur_node.children.append(node)
            cur_node = node
            # print('idx:', cur_node.idx, 'label:', cur_node.label)
            idx += 1
    # print('----------end root-----------')
    # print(all_nodes_new)
    # print('---------------------')
    # print('children:', len(root.children))
    return root, all_nodes_new

def get_typ_idx(strr):
    if ':' in strr:
        typ, name_idx = strr.split(':')
        typ = typ.strip()
        name_idx = name_idx.strip()
        name = name_idx[:name_idx.index('(')]
        idx = name_idx[name_idx.index('('):]
        idx = idx.lstrip('(').rstrip(')')
        return ActionNode(typ, int(idx), name)

    else:    
        typ = strr[:strr.index('(')]
        idx = strr[strr.index('('):]
        idx = idx.lstrip('(').rstrip(')')
        if typ == 'NullLiteral':
            return ActionNode(typ, int(idx), 'null')
        if typ == 'ThisExpression':
            return ActionNode(typ, int(idx), 'this')
        return ActionNode(typ, int(idx))

def get_typ_pos(strr):
    if ':' in strr:
        typ, name_pos = strr.split(':',1)
        typ = typ.strip()
        name_pos = name_pos.strip()
        label = name_pos[:name_pos.index('[')].strip()
        pos = name_pos[name_pos.index('['):]
        # idx = idx.lstrip('(').rstrip(')')
        # return ActionNode(typ, int(idx), label)
        return ActionNode1(typ, pos, label)

    else:
        typ = strr[:strr.index('[')].strip()
        pos = strr[strr.index('['):]
        if typ == 'NullLiteral':
            return ActionNode1(typ, pos, 'null')
        if typ == 'ThisExpression':
            return ActionNode1(typ, pos, 'this')
        return ActionNode1(typ, pos)

def parse_action_string(actions_string):
    actions = actions_string.strip().split('===')
    # print(actions)
    # print('parse_action')
    new_actions = []
    try:
        for action in actions[1:]:
            action = action.strip()
            # print('action:')
            # print(action)
            type, nodes = action.split('\n---\n')
            # print('action1:')
            # print(type, nodes,type.split('\n'))
            new_actions.append({"type": type.split('\n')[0].strip(), "nodes": nodes.strip()})
            # break


    except Exception as e:
        print('273:',e)

    # print(new_actions)
    return new_actions

def get_ast_action(file_name1, file_name2, root1, root2, tem_path):
    # print('----------start ast action-----------')
    tem_path = './data/diff_utils' 
    tem_path1 = '../gumtree/gumtree'
    # out = subprocess.Popen('C:/Users/wang/Desktop/gumtree-3.0.0/bin/gumtree diff %s/%s.java %s/%s.java'%(tem_path, file_name1, tem_path, file_name2), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    out = subprocess.Popen('./data/diff_utils/gumtree/gumtree-3.0.0/bin/gumtree textdiff %s/%s.java %s/%s.java'%(tem_path, file_name1, tem_path, file_name2), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    # out = subprocess.Popen('C:/Users/wang/Desktop/gumtree-3.0.0/bin/gumtree textdiff %s/%s.java %s/%s.java'%(tem_path, file_name1, tem_path, file_name2), shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    

    try:
        # print('hahaha')
        stdout = stdout.decode('utf-8')
        # print(stdout)
        # for x in stdout.splitlines():
        #     print('--------')
        #     print(x.strip())
        #     break
    except Exception as e:
        print("295:",e)

    actions = parse_action_string(stdout)

    # raw_actions = [x.strip() for x in stdout.splitlines() if x.strip()]

    # all_match = []
    # all_delete = []
    # all_update = []
    # all_move = []
    # all_add = []

    all_old = []
    all_new = []
    try:
        for i, action in enumerate(actions):
            # print('aaa')
            # print(action)
            # if i < 3:

                # break
            action_type = action['type']
            if action_type == 'match':
                continue

            elif action_type == 'move-tree':
                # continue
                # print(action)       
                raw_node_old, tem = action['nodes'].split('\nto\n')
                raw_node_new, _ = tem.rsplit('at', 1)
                raw_node_old = raw_node_old.strip().split('\n')
                raw_node_new = raw_node_new.strip().split('\n')

                # print('see tree struct')
                for node in raw_node_new:
                    # print(node.strip())
                    node_new = get_typ_pos(node.strip())
                    if len(all_new) > 0 and all_new[-1] == node_new:
                        continue
                    all_new.append(node_new)
                for node in raw_node_old:
                    # print(node.strip())
                    node_old = get_typ_pos(node.strip())
                    if len(all_old) > 0 and all_old[-1] == node_old:
                        continue
                    all_old.append(node_old)

            elif action_type == 'delete-tree':
                # continue
                # print(action)
                # print('delete:',action['nodes'])
                raw_node_old = action['nodes']
                raw_node_old = raw_node_old.strip().split('\n')

                for node in raw_node_old:
                    # print(node.strip())
                    node_old = get_typ_pos(node.strip())
                    if len(all_old) > 0 and all_old[-1] == node_old:
                        continue
                    all_old.append(node_old)

            elif action_type == 'delete-node':
                # continue
                # print(action)
                raw_node_old = action['nodes']
                raw_node_old = raw_node_old.strip()
                # pos = pos.strip()
                node_old = get_typ_pos(raw_node_old)
                if len(all_old) > 0 and all_old[-1] == node_old:
                        continue
                all_old.append(node_old)

            elif action_type == 'insert-node':
                # continue
                # print(action)
                raw_node_new, tem = action['nodes'].split('\nto\n')
                raw_node_par, _ = tem.rsplit('at',1)
                raw_node_new = raw_node_new.strip()
                raw_node_par = raw_node_par.strip()
                # pos = pos.strip()
                node_new = get_typ_pos(raw_node_new)
                node_old = get_typ_pos(raw_node_par)
                if len(all_old) > 0 and all_old[-1] != node_old:
                    all_old.append(node_old)
                if len(all_new) > 0 and  all_new[-1] != node_new:
                    all_new.append(node_new)
                # all_old.append(node_old)
                # all_new.append(node_new)


            elif action_type == 'insert-tree':
                # print(action)
                raw_node_new, tem = action['nodes'].split('\nto\n')
                raw_node_par, _ = tem.rsplit('at',1)
                raw_node_new = raw_node_new.strip().split('\n')
                raw_node_par = raw_node_par.strip().split('\n')
                # print('see tree struct')
                for node in raw_node_new:
                    # print(node.strip())
                    node_new = get_typ_pos(node.strip())
                    if len(all_new) > 0 and all_new[-1] == node_new:
                        continue
                    all_new.append(node_new)
                for node in raw_node_par:
                    # print(node.strip())
                    node_old = get_typ_pos(node.strip())
                    if len(all_old) > 0 and  all_old[-1] == node_old:
                        continue
                    all_old.append(node_old)

            elif action_type == 'update-node':
                # print(action)
                raw_node_old, tem = action['nodes'].split('\n')
                # print(raw_node_old,'\n', tem)
                node_old = get_typ_pos(raw_node_old)
                # all_update.append(node_old)

                if len(all_old) > 0 and all_old[-1] == node_old:
                        continue
                all_old.append(node_old)
                # continue

            else:
                # print('is else')
                # print(action)
                continue

    except Exception as e:
        print("407",e)
        # continue
    # print('old:',all_old)
    # print('------------')
    # print('new:',all_new)
    # print('------------')
    # all_old = list(set(all_old))
    # all_new = list(set(all_new))
    # print('-----end ast action-------')

    return all_old, all_new


def get_ast_diff_tokens(old_nodes, new_nodes, all_delete, all_update, all_move, all_add):
    asts = []
    ast_diffs = []
    i = 0
    cnt = 0

    # print(len(old_nodes), len(all_delete), len(all_update), len(all_move), len(all_add))
    # print('-----ast diff-----')
    for node in old_nodes:
        flag = False

        for j in range(len(all_update)):
            cur_upd = all_update[j]
            # print(i, cur_upd.pos, cur_upd.typ)
            if cur_upd.pos == node.pos and cur_upd.typ == node.typeLabel:
                flag = True
                # print('----------get update--------')

        for j in range(len(all_move)):
            cur_mov = all_move[j][0]
            # print(i, cur_mov.pos, cur_mov.typ)
            if cur_mov.pos == node.pos and cur_mov.typ == node.typeLabel:
                flag = True
                # print('----------get move--------')

        for j in range(len(all_add)):
            cur_add = all_add[j][0]
            # print(i, cur_add.pos, cur_add.typ)
            if cur_add.pos == node.pos and cur_add.typ == node.typeLabel:
                flag = True
                # print('----------get add--------')
        
        asts.append(node.typeLabel)
        if flag:
            # ast_diffs += [1] * len(typeLabel)
            ast_diffs += [1]
            cnt += 1
        else:
            ast_diffs += [0]
        i += 1
        # print('\n')
        # if i == 10:
        #     break

    # print('----res------', cnt)
    # for j in range(len(all_delete)):
    #     cur_del = all_delete[j]
    #     print(cur_del.pos, cur_del.typ)
    # for j in range(len(all_update)):
    #     cur_del = all_update[j]
    #     print(cur_del.pos, cur_del.typ)
    # for j in range(len(all_move)):
    #     cur_del = all_move[j][0]
    #     print(cur_del.pos, cur_del.typ)
    # for j in range(len(all_add)):
    #     cur_del = all_add[j][0]
    #     print(cur_del.pos, cur_del.typ)

    # print('\n')
    # print('---ast--------')
    # for node in old_nodes:
    #     print(node.pos, node.typeLabel, node.label)
    return asts, ast_diffs

def read_csv_files(file, tem_path):
    all_deletes = []
    all_updates = [] 
    all_moves = []
    all_adds = []
    all_old_nodes = []
    all_new_nodes = []

    with open(file, 'r' , encoding='utf-8') as f:
        # j = 0
        k = 0
        for line in f:
            k += 1
            # if k < 73:
            #     continue

            print('-------->当前处理的数据是第：', k)
            line = line.strip().split('\t')
            before = line[0]
            after = line[1]

            flag = ''

            #处理before_code，最终code中含comment所指的区域，<START> <END>表示
            before_code, comment = before.split('</code>', 1)
            before_code = before_code[6:]
            comment = ''.join(comment.split('</technical_language>'))[20:]

            try:
                if "<START>" in before_code:
                    str1, str2 = before_code.split("<START>")
                    if "<END>" in str2:
                        str2_1, str2_2 = str2.split("<END>")
                    else:
                        str2_1, str2_2 = str2.split("END>")
                        flag = "--------> flase_<END>"
                    before_code_line = str1 + str2_1 + str2_2
                else:
                    before_code_line = before_code
                    flag = "--------> flase_<START>"
            except Exception as e:
                continue

            before_path = os.path.join(tem_path, "before.java")
            after_path = os.path.join(tem_path, "after.java")
            with open(before_path, 'w', encoding='utf-8') as f1:
                f1.write("public class Test {" + before_code_line + "}")
            with open(after_path, 'w', encoding='utf-8') as f1:
                f1.write("public class Test {" + after + "}")

            # break

            root_old, old_nodes = get_ast_root(tem_path, 'before')
            root_new, new_nodes = get_ast_root(tem_path, 'after')
            all_delete, all_update = get_ast_action('before', 'after', root_old, root_new, tem_path)

            all_old_nodes.append(old_nodes)
            all_new_nodes.append(new_nodes)
            all_deletes.append(all_delete)   # for old ast change
            all_updates.append(all_update)   # for new ast change
            # all_moves.append(all_move)
            # all_adds.append(all_add)

            # for node in old_nodes:
            #     print(node.pos, node.typeLabel, node.label)
            # print('----res after get action------','\n')
            # for j in range(len(all_delete)):
            #     cur_del = all_delete[j]
            #     print(cur_del.pos, cur_del.typ)
            # print('-----end delete------')
            # for j in range(len(all_update)):
            #     cur_del = all_update[j]
            #     print(cur_del.pos, cur_del.typ)
            # print('-----end update------')

            # for j in range(len(all_move)):
            #     cur_del = all_move[j][0]
            #     print(cur_del.pos, cur_del.typ)
            # print('-----end move------')
            # for j in range(len(all_add)):
            #     cur_del = all_add[j][0]
            #     print(cur_del.pos, cur_del.typ)
            # print('-----end add------')
            # if k == 74:
            #     break

    # for nodes in all_old_nodes:
    #     print('len is:', len(nodes))
    #     for node in nodes:
    #         print(node.pos, node.typeLabel, node.label)
    #     print('\n')

    # print('----res after get action------','\n')
    # for j in range(len(all_deletes)):
    #     cur_dels = all_deletes[j]
    #     print('len is:', len(all_deletes))
    #     for cur_del in cur_dels:
    #         print(cur_del.pos, cur_del.typ)
    # print('-----end delete------','\n')
    # for j in range(len(all_updates)):
    #     cur_upads = all_updates[j]
    #     print('len is:', len(all_updates))
    #     for cur_upa in cur_upads:
    #         print(cur_upa.pos, cur_upa.typ)
    # print('-----end update------', '\n')

    # json.dump(all_deletes, open('dataset/all_deletes.json', 'w'))
    # json.dump(all_updates, open('dataset/all_updates.json', 'w'))
    # json.dump(all_moves, open('dataset/all_moves.json', 'w'))
    # json.dump(all_adds, open('dataset/all_adds.json', 'w'))

    pickle.dump(all_deletes, open('dataset/all_deletes.pkl', 'wb'))
    pickle.dump(all_updates, open('dataset/all_updates.pkl', 'wb'))
    pickle.dump(all_old_nodes, open('dataset/all_old_nodes.pkl', 'wb'))
    pickle.dump(all_new_nodes, open('dataset/all_new_nodes.pkl', 'wb'))

    return 

def read_filter_pkl_files():
    path = os.path.join(os.path.realpath("."))
    all_old_nodes = pickle.load(open(path+'/dataset/large_ast/train/all_old_nodes.pkl', 'rb')) 
    all_new_nodes = pickle.load(open(path+'/dataset/large_ast/train/all_new_nodes.pkl', 'rb')) 
    all_deletes = pickle.load(open(path+'/dataset/large_ast/train/all_deletes.pkl', 'rb')) 
    all_updates = pickle.load(open(path+'/dataset/large_ast/train/all_updates.pkl', 'rb')) 

    all_old_ast_tokens = []
    all_old_ast_maps = []
    all_new_ast_tokens = []
    all_new_ast_maps = []

    print(len(all_old_nodes), len(all_new_nodes), len(all_deletes), len(all_updates))

    
    # for nodes in all_old_nodes:
    #     print('len is:', len(nodes))
    #     for node in nodes:
    #         print(node.pos, node.typeLabel, node.label)
    #     print('\n')

    # print('----res after get action------','\n')
    # for j in range(len(all_deletes)):
    #     cur_dels = all_deletes[j]
    #     print('len is:', len(all_deletes))
    #     for cur_del in cur_dels:
    #         print(cur_del.pos, cur_del.typ)
    # print('-----end delete------','\n')
    # for j in range(len(all_updates)):
    #     cur_upads = all_updates[j]
    #     print('len is:', len(all_updates))
    #     for cur_upa in cur_upads:
    #         print(cur_upa.pos, cur_upa.typ)
    # print('-----end update------', '\n')

    for i in range(len(all_old_nodes)):
        old_ast_tokens = []
        old_ast_maps = []
        new_ast_tokens = []
        new_ast_map = []

        nodes = all_old_nodes[i]
        cur_dels = all_deletes[i]
        cur_updas = all_updates[i]
        old_ast_tokens, old_ast_maps = get_ast_sequence(nodes, cur_dels)
        new_ast_tokens, new_ast_maps = get_ast_sequence(nodes, cur_updas)

        all_old_ast_tokens.append(old_ast_tokens)
        all_new_ast_tokens.append(new_ast_tokens)
        all_old_ast_maps.append(old_ast_maps)
        all_new_ast_maps.append(new_ast_maps)

        # if i == 2:
        #     break

    print(len(all_old_ast_tokens), len(all_old_ast_maps), len(all_new_ast_tokens), len(all_new_ast_maps))
    pickle.dump(all_old_ast_tokens, open(path+'/dataset/dataset_saved/large_ast/train_tokens/all_old_ast_tokens.pkl', 'wb'))
    pickle.dump(all_old_ast_maps, open(path+'/dataset/dataset_saved/large_ast/train_tokens/all_old_ast_maps.pkl', 'wb'))
    pickle.dump(all_new_ast_tokens, open(path+'/dataset/dataset_saved/large_ast/train_tokens/all_new_ast_tokens.pkl', 'wb'))
    pickle.dump(all_new_ast_maps, open(path+'/dataset/dataset_saved/large_ast/train_tokens/all_new_ast_maps.pkl', 'wb'))
    return  all_old_ast_tokens, all_old_ast_maps, all_new_ast_tokens, all_new_ast_maps

def get_ast_sequence(nodes, all_changes):
    asts = []
    asts_map_changes = []

    for node in nodes:
        flag = False
        if node.label == 'root':
            continue
        # if node.
        
        for j in range(len(all_changes)):
            cur_change = all_changes[j]
            if cur_change.pos == node.pos and cur_change.typ == node.typeLabel:
                flag = True
                break
        
        tokens_ori = tokenizer.tokenize(node.typeLabel)

        asts += tokens_ori
        if flag:
            asts_map_changes += [1] * len(tokens_ori)
        else:
            asts_map_changes += [0] * len(tokens_ori)

    return asts, asts_map_changes

def merge_pkl_file():
    all_old_nodes_0 = pickle.load(open('dataset/train0/all_old_nodes.pkl', 'rb')) 
    all_old_nodes_1 = pickle.load(open('dataset/train1/all_old_nodes.pkl', 'rb')) 
    all_old_nodes_0 += all_old_nodes_1
    pickle.dump(all_old_nodes_0, open('dataset/train/all_old_nodes.pkl', 'wb'))

    all_new_nodes_0 = pickle.load(open('dataset/train0/all_new_nodes.pkl', 'rb')) 
    all_new_nodes_1 = pickle.load(open('dataset/train1/all_new_nodes.pkl', 'rb')) 
    all_new_nodes_0 += all_new_nodes_1
    pickle.dump(all_new_nodes_0, open('dataset/train/all_new_nodes.pkl', 'wb'))

    all_deletes_0 = pickle.load(open('dataset/train0/all_deletes.pkl', 'rb')) 
    all_deletes_1 = pickle.load(open('dataset/train1/all_deletes.pkl', 'rb')) 
    all_deletes_0 += all_deletes_1
    pickle.dump(all_deletes_0, open('dataset/train/all_deletes.pkl', 'wb'))

    all_updates_0 = pickle.load(open('dataset/train0/all_updates.pkl', 'rb')) 
    all_updates_1 = pickle.load(open('dataset/train1/all_updates.pkl', 'rb')) 
    all_updates_0 += all_updates_1
    pickle.dump(all_old_nodes_0, open('dataset/train/all_updates.pkl', 'wb'))

if __name__ == '__main__':
    tem_path = '../gumtree/gumtree/bin' 
    file_path = 'larger_filter/train.tsv'
    # pkl_file_path = ''
    # read_csv_files(file_path, tem_path)

    # print(os.path.list())
    read_filter_pkl_files()

    # merge_pkl_file()

    # get_ast_diff_tokens(old_nodes, new_nodes , all_delete, all_update, all_move, all_add )
    # get_ast_action('before', 'after', None, None, tem_path)
