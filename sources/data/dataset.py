import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset

import os
import random
import logging
import pickle

import enums
from .data_utils import load_dataset_from_dir

logger = logging.getLogger(__name__)

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

    # all_deletes = []
    # all_updates = [] 
    # # all_moves = []
    # # all_adds = []
    # all_old_nodes = []
    # all_new_nodes = []

    all_old_ast_tokens = []
    all_old_ast_maps = []
    all_new_ast_tokens = []
    all_new_ast_maps = []

    # global k

    with open(file, 'rb') as f:
        j = 0
        k = 0
        obj = pickle.load(f)
        length = len(obj[0])
        # for i in range(int(length * 0.9), length):
        # for i in range(int(length * 0.8)):
        for i in range(length):
            k += 1
            # if (k < 16722):
            if (k < 18812):
                continue
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

            # result = fileLineDiff(before_split_code,after_split_code)

            # print('\n\n\n')
            if (not check_tokenize_line(before_split_code)) or (not check_tokenize_line(after_split_code)):
                # print("error split")
                continue 

            code_diff = []
            code_tokens = []
            diff_line = []

            # try:
            #     for elem in result:
            #         # print('elem:', elem[0].strip())
            #         if isinstance(elem, Keep ):
            #             code_tokens.append('KEEP '+ elem[0].strip() + '\n')
            #             diff_line.append(0)
            #         elif isinstance(elem, Insert):
            #             code_tokens.append('ADD ' + elem[0].strip() + '\n')
            #             diff_line.append(1)
            #         elif isinstance(elem, Remove):
            #             code_tokens.append('DEL ' + elem[0].strip() + '\n')
            #             diff_line.append(2)
            # except Exception as e:
            #     continue

            # code_diff = ' '.join(code_tokens)

            # print('code diff is: ---\n', code_diff)
            # print('before_code_line is: ---\n', before_code_line)
            # print('after is:---\n', after)

            # before_path = os.path.join(os.path.realpath("."), "data/diff_utils/before.java")
            before_path = "before.java"
            after_path = "after.java"
            with open(before_path, 'w', encoding='utf-8') as f1:
                f1.write("public class Test {" + before_code_line + "}")
            with open(after_path, 'w', encoding='utf-8') as f1:
                f1.write("public class Test {" + after + "}")

            # get ast root
            # print('-----start get ast root------')
            tem_path = '../gumtree/gumtree/bin' 
            root_old, old_nodes = get_ast_root(tem_path, 'before')
            root_new, new_nodes = get_ast_root(tem_path, 'after')
            all_delete, all_update = get_ast_action('before', 'after', root_old, root_new, tem_path)

            # all_old_nodes.append(old_nodes)
            # all_new_nodes.append(new_nodes)
            # all_deletes.append(all_delete)   # for old ast change
            # all_updates.append(all_update)   # for new ast change
            # for i in range(len(all_old_nodes)):
            old_ast_tokens = []
            old_ast_maps = []
            new_ast_tokens = []
            new_ast_maps = []

            nodes = old_nodes
            cur_dels = all_delete
            cur_updas = all_update
            old_ast_tokens, old_ast_maps = get_ast_sequence(nodes, cur_dels)
            new_ast_tokens, new_ast_maps = get_ast_sequence(nodes, cur_updas)

            all_old_ast_tokens.append(old_ast_tokens)
            all_new_ast_tokens.append(new_ast_tokens)
            all_old_ast_maps.append(old_ast_maps)
            all_new_ast_maps.append(new_ast_maps)



            # for node in old_nodes:
            #     print(node.pos, node.typeLabel, node.label)
            #     print('----res after get action------','\n')
            #     for j in range(len(all_delete)):
            #         cur_del = all_delete[j]
            #         print(cur_del.pos, cur_del.typ)
            #     print('-----end delete------')
            #     for j in range(len(all_update)):
            #         cur_del = all_update[j]
            #         print(cur_del.pos, cur_del.typ)
            #     print('-----end update------')

            # ast_diff = []
            # ast_diff = get_ast_diff()
            # ast_diffs.append(ast_diff)

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
            # if j == 10:
            #     break
        logger.info(f"当前的数据集是： {file}")
        logger.info(f"当前处理的数据共：{j}")
        print("当前处理的数据共：",j)
        # print(len(all_old_nodes), len(all_new_nodes), len(all_deletes), len(all_updates))


        # for i in range(len(all_old_ast_tokens)):
        #     print(all_old_ast_tokens[i])
        #     print(all_old_ast_maps[i])
        #     print(all_new_ast_tokens[i])
        #     print(all_new_ast_maps[i])
        #     nodes = all_old_nodes[i]
        #     cur_dels = all_deletes[i]
        #     cur_updas = all_updates[i]
        #     old_ast_tokens, old_ast_maps = get_ast_sequence(nodes, cur_dels)
        #     new_ast_tokens, new_ast_maps = get_ast_sequence(nodes, cur_updas)

        #     all_old_ast_tokens.append(old_ast_tokens)
        #     all_new_ast_tokens.append(new_ast_tokens)
        #     all_old_ast_maps.append(old_ast_maps)
        #     all_new_ast_maps.append(new_ast_maps)
        print(len(new_code_diffs), len(docs), len(labels), len(lines_level))
        print(len(all_old_ast_tokens), len(all_old_ast_maps), len(all_new_ast_tokens), len(all_new_ast_maps))
        pickle.dump(all_old_ast_tokens, open('./accumulo/test_tokens/all_old_ast_tokens.pkl', 'wb'))
        pickle.dump(all_old_ast_maps, open('./accumulo/test_tokens/all_old_ast_maps.pkl', 'wb'))
        pickle.dump(all_new_ast_tokens, open('./accumulo/test_tokens/all_new_ast_tokens.pkl', 'wb'))
        pickle.dump(all_new_ast_maps, open('./accumulo/test_tokens/all_new_ast_maps.pkl', 'wb'))
        
        # read_filter_pkl_files()
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
    return new_code_diffs, docs, labels, code_befores, code_afters, all_code_tokens, lines_level, tokens_level


class CodeDataset(Dataset):

    def __init__(self, mode, args=None,  task=None, language=None, stage=None, dataset_size=None):
        
        super(CodeDataset, self).__init__()
        self.args = args
        self.task = task
        self.mode = mode
        self.stage = stage
        self.dataset_size = dataset_size

        #dataset dir: fine_tune or pre_train
        # self.dataset_dir = os.path.join(args.dataset_root, self.mode)

        self.code_diffs, self.docs, self.labels, self.code_befores, self.code_afters,  self.code_tokens, self.lines_level, self.tokens_level = get_code_ast_diff('accumulo/accumulo.pkl')
        self.size = len(self.code_diffs)

        # if self.mode == 'pre_train':
        #     self.code_diffs, self.ast_diffs, self.docs, self.code_befores, self.code_afters,  self.code_tokens, self.lines_level, self.tokens_level = load_dataset_from_dir(dataset_dir=self.dataset_dir,stage='train')
        #     self.size = len(self.code_diffs)

        # # load fine-tuning dataset
        # else:
        #     self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_size)
        #     self.code_diffs, self.ast_diffs, self.docs, self.code_befores, self.code_afters, self.code_tokens, self.lines_level, self.tokens_level = load_dataset_from_dir(dataset_dir=self.dataset_dir,stage=self.stage)
        #     self.size = len(self.code_diffs)

    
    def __getitem__(self, index):
        return self.code_diffs[index], self.docs[index], self.labels[index],self.code_tokens[index], self.code_befores[index], self.code_afters[index], self.lines_level[index], self.tokens_level[index]

    def __len__(self):
        return self.size

    def set_task(self, task):
        self.task = task

    def save(self):
        """Save to binary pickle file"""
        # if self.mode == 'pre_train':
        #     path_root = os.path.join(self.args.dataset_save_dir, self.mode)
        # else:
        #     path_root = os.path.join(self.args.dataset_save_dir, self.mode, self.dataset_size)
        # path = os.path.join(path_root, f'{self.stage}.pk')  

        path = os.path.join('./accumulo/test_tokens/','test.pk')

        with open(path, mode='wb') as f:
            pickle.dump(self, f)
        
def init_dataset(args, mode, task=None, language=None, stage=None, load_if_saved=True) -> CodeDataset:
    if load_if_saved:
        # if mode == 'fine_tune':
        #     path_root = os.path.join(args.dataset_save_dir, mode, args.dataset_size)
        #     path = os.path.join(path_root, f'{stage}.pk')
        # else:
        #     path = os.path.join(args.dataset_save_dir, mode, f'{stage}.pk')
        # path = os.path.join(args.dataset_save_dir, 'accumulo', stage+'_tokens', f'{stage}.pk')
        # if os.path.exists(path) and os.path.isfile(path):
        #     logger.info(f'Trying to load saved binary pickle file from: {path}')
        #     with open(path, mode='rb') as f:
        #         obj = pickle.load(f)
        #     assert isinstance(obj, CodeDataset)
        #     obj.args = args
        #     logger.info(f'Dataset instance loaded from: {path}')
        #     return obj


                # logger.info('get fine_tune')
        path_root = os.path.join(args.dataset_save_dir, 'accumulo', stage+'_tokens')

        code_diffs_path = os.path.join(path_root, 'code_diffs.pkl')
        docs_path = os.path.join(path_root, 'docs.pkl')
        labels_path = os.path.join(path_root, 'labels.pkl')
        # code_befores_path = os.path.join(path_root, 'code_befores.pkl')
        # code_afters_path = os.path.join(path_root, 'code_afters.pkl')
        lines_level_path = os.path.join(path_root, 'lines_level.pkl')
        tokens_level_path = os.path.join(path_root, 'tokens_level.pkl')
    # else:
    #     path = os.path.join(args.dataset_save_dir, mode, f'{stage}.pk')
    # if os.path.exists(path_root) and os.path.isfile(path_root):
        logger.info(f'Trying to load saved ast binary pickle file from: {path_root}')
        code_diffs = pickle.load(open(code_diffs_path, 'rb')) 
        docs = pickle.load(open(docs_path, 'rb')) 
        labels = pickle.load(open(labels_path, 'rb')) 
        # all_old_maps = pickle.load(open(code_befores_path, 'rb')) 
        # all_old_maps = pickle.load(open(code_afters_path, 'rb')) 
        lines_level = pickle.load(open(lines_level_path, 'rb')) 
        tokens_level = pickle.load(open(tokens_level_path, 'rb')) 

        # with open(path, mode='rb') as f:
        #     obj = pickle.load(f)
        # assert isinstance(obj, CodeDataset)
        # obj.args = args
        # logger.info(f'Dataset instance loaded from: {path}')
        # logger.info(f'dataset:{len(all_new_nodes)}')
        return CodeDiffDataset(
            code_diffs,
            docs,
            labels,
            lines_level,
            tokens_level,
            len(code_diffs)
        )
    
    # dataset = CodeDataset(args=args,
    #                       mode=mode,
    #                       task=task,
    #                       language=language,
    #                       stage=stage,
    #                       dataset_size=args.dataset_size)
    # dataset.save()

    # return dataset

class AstDiffDataset(object):
    """A single training/test features for a example."""
    def __init__(self,
            all_new_nodes,
            all_old_nodes,
            all_new_maps,
            all_old_maps,
            size
    ):
        self.all_new_nodes = all_new_nodes
        self.all_old_nodes = all_old_nodes
        self.all_new_maps = all_new_maps
        self.all_old_maps = all_old_maps
        self.size = size
class CodeDiffDataset(object):
    """A single training/test features for a example."""
    def __init__(self,
            code_diffs,
            docs,
            labels,
            lines_level,
            tokens_level,
            size
    ):
        self.code_diffs = code_diffs
        self.docs = docs
        self.labels = labels
        self.lines_level = lines_level
        self.tokens_level = tokens_level
        self.size = size

def init_ast_dataset(args, mode, task=None, stage=None, load_if_saved=True):
    if load_if_saved:
        if mode == 'fine_tune':
            # logger.info('get fine_tune')
            path_root = os.path.join(args.dataset_save_dir, 'accumulo', stage+'_tokens')

            new_ast_path = os.path.join(path_root, 'all_new_ast_tokens.pkl')
            old_ast_path = os.path.join(path_root, 'all_old_ast_tokens.pkl')
            new_maps_path = os.path.join(path_root, 'all_new_ast_maps.pkl')
            old_maps_path = os.path.join(path_root, 'all_old_ast_maps.pkl')
        # else:
        #     path = os.path.join(args.dataset_save_dir, mode, f'{stage}.pk')
        # if os.path.exists(path_root) and os.path.isfile(path_root):
            logger.info(f'Trying to load saved ast binary pickle file from: {path_root}')
            all_new_nodes = pickle.load(open(new_ast_path, 'rb')) 
            all_old_nodes = pickle.load(open(old_ast_path, 'rb')) 
            all_new_maps = pickle.load(open(new_maps_path, 'rb')) 
            all_old_maps = pickle.load(open(old_maps_path, 'rb')) 

            # with open(path, mode='rb') as f:
            #     obj = pickle.load(f)
            # assert isinstance(obj, CodeDataset)
            # obj.args = args
            # logger.info(f'Dataset instance loaded from: {path}')
            # logger.info(f'dataset:{len(all_new_nodes)}')
            return AstDiffDataset(
                all_new_nodes,
                all_old_nodes,
                all_new_maps,
                all_old_maps,
                len(all_new_nodes)
            )

class TextDataset(Dataset):
    def __init__(self, examples, args, tokenizer, task=None):
        self.examples = examples
        self.args = args
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self):
        return len(self.examples)

    def set_task(self, task):
        self.task = task

    def __getitem__(self, item):
        input_ids = []
        input_tokens = []
                             
        if self.task == enums.TASK_TAG_PREDICTION:  
            index = self.examples[item].source_tokens.index(self.tokenizer.sep_token,4)
            #Code_Diff Tag Prediction
            for idx, i in enumerate(self.examples[item].source_tokens[:index+1]):
                if i in ["KEEP", "ADD","DEL"]:
                    input_tokens.append(self.tokenizer.mask_token)
                else:
                    input_tokens.append(i)
            #AST_Diff Tag Prediction
            for idx, i in enumerate(self.examples[item].source_ast_tokens):
                i = i.replace('\u0120','')
                if i in ['Insert','Move','Delete','Update']:
                    input_tokens.append(self.tokenizer.mask_token)
                else:
                    input_tokens.append(i)
            if len(input_tokens) < len(self.examples[item].source_ids):
                padding_length = len(self.examples[item].source_ids) - len(input_tokens)
                input_tokens += [self.tokenizer.pad_token] * padding_length
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            return (torch.tensor(input_ids),                         
                    torch.tensor(self.examples[item].source_ids)          
                    )
        elif self.task == enums.TASK_COMMNET_GENERATION:
            return (torch.tensor(self.examples[item].source_ids),     
                    torch.tensor(self.examples[item].target_ids),
                    torch.tensor(self.examples[item].labels),
                    #  for diffrent level 
                    torch.tensor(self.examples[item].line_level_code),
                    torch.tensor(self.examples[item].token_level_code),
                    # for ast
                    torch.tensor(self.examples[item].old_ast_ids),
                    torch.tensor(self.examples[item].new_ast_ids), 
                    torch.tensor(self.examples[item].old_ast_maps),
                    torch.tensor(self.examples[item].new_ast_maps), 
                    )