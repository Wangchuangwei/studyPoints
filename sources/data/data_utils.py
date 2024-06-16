import os 
import re
import logging
from tqdm import tqdm
import pandas as pd

# from .diff_utils.get_diff import get_code_ast_diff
from .diff_utils.get_new_data import get_code_ast_diff

logger = logging.getLogger(__name__)

def parse_tsv_file(file):
    path = os.path.join(os.path.realpath("."), "data/diff_utils/accumulo/accumulo.pkl")
    code_diffs, ast_diffs, docs, labels, code_befores, code_afters, code_tokens, lines_level, tokens_level = get_code_ast_diff(path)
    # code_diffs, ast_diffs, docs, code_befores, code_afters, code_tokens, lines_level, tokens_level = get_code_ast_diff(file)
    return code_diffs, ast_diffs, docs, labels, code_befores, code_afters, code_tokens, lines_level, tokens_level  

def iter_all_files(base): 
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield os.path.join(root, f)

def iter_per_train_dataset_files(dataset_dir,stage):
    """
    Get files for pre-training, all files with extension ``tsv`` will be included.

    """
    # return [file for file in iter_all_files(base=dataset_dir) if file.endswith('.tsv')]
    return [file for file in iter_all_files(base=dataset_dir) if file.endswith(stage+'.tsv')]

def load_pre_train_dataset(file):
    """
    Load tsv dataset from given file

    """

    code_diffs, ast_diffs, docs, labels, code_befores, code_afters, code_tokens, lines_level, tokens_level  = parse_tsv_file(file)
    return code_diffs, ast_diffs, docs, labels, code_befores, code_afters, code_tokens, lines_level, tokens_level 

def load_dataset_from_dir(dataset_dir,stage):
    """
    Load all files in the given dir, only for pre-training

    """

    all_code_diffs = []
    all_ast_diffs = []
    all_docs = []
    all_code_tokens = []
    all_code_befores = []
    all_code_afters = []
    all_line_level = []
    all_token_level = []

    all_labels = []

    if stage is not None:
        dataset_files = iter_per_train_dataset_files(dataset_dir,stage)
    else:
        dataset_files = iter_per_train_dataset_files(dataset_dir,"")
    logger.info(f"dataset_files_path:{dataset_files}")
    if len(dataset_files) > 0:
        n_sample = 0
        for dataset_file_path in dataset_files:
            code_diffs, ast_diffs, docs, labels, code_befores, code_afters, code_tokens, lines_level, tokens_level = load_pre_train_dataset(file=dataset_file_path) 
            all_code_diffs += code_diffs
            all_ast_diffs += ast_diffs
            all_docs += docs
            all_code_tokens += code_tokens
            all_code_befores += code_befores
            all_code_afters += code_afters
            all_line_level += lines_level
            all_token_level += tokens_level

            all_labels += labels

            n_line = len(code_diffs)
            n_sample += n_line
            logger.info(f'    File: {dataset_file_path}, {n_line} samples')
            
            break

        logger.info(f' dataset size: {n_sample}')

    logger.info(f"{len(all_code_diffs)}is{len(all_ast_diffs)} is{len(all_docs)}")
    assert len(all_code_diffs) == len(all_ast_diffs) == len(all_docs) == len(all_code_tokens)
    return all_code_diffs, all_ast_diffs, all_docs, all_labels, all_code_tokens, all_code_befores, all_code_afters, all_line_level, all_token_level

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
            source_tokens,
            source_ids,
            source_mask,
            target_tokens,
            target_ids,
            labels,
            line_level_code,
            token_level_code,
            old_ast_ids,
            old_ast_mask,
            new_ast_ids,
            new_ast_mask,
            old_ast_maps,
            new_ast_maps
    ):
        self.source_tokens = source_tokens
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.target_tokens = target_tokens
        self.target_ids = target_ids
        self.labels = labels
        self.line_level_code = line_level_code
        self.token_level_code = token_level_code

        self.old_ast_ids = old_ast_ids
        self.new_ast_ids = new_ast_ids
        self.old_ast_mask = old_ast_mask
        self.new_ast_mask = new_ast_mask
        self.old_ast_maps = old_ast_maps
        self.new_ast_maps = new_ast_maps

def convert_examples_to_features(examples, tokenizer, args, size
                                , stage=None, ast_changes=None
                                ):
    """convert examples to token ids"""

    features = []
    for i in range(size):
        #source
        if args.model_type == 'codet5':
            # source_code_tokens = tokenizer.tokenize(examples.code_diffs[i])
            # ast_diffs = " ".join(examples.ast_diffs[i])
            # source_ast_tokens = tokenizer.tokenize(ast_diffs)

            # source_tokens = source_ast_tokens[:args.ast_length-2]
            # source_tokens = [tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
            # source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

            # source_code_tokens = source_ast_tokens[:args.code_length-1]
            # source_code_tokens = source_code_tokens+[tokenizer.sep_token]

            # source_tokens += [x for x in source_code_tokens]
            # source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_code_tokens]
            
            # source_mask = [1] * (len(source_tokens))
            # padding_length = args.code_length + args.ast_length - len(source_ids)
            # # position_idx += [tokenizer.pad_token_id] * padding_length
            # source_ids += [tokenizer.pad_token_id] * padding_length
            # source_mask+=[0]*padding_length 
            
            # #target
            # if stage == 'test':
            #     target_tokens = tokenizer.tokenize("None")
            # else:
            #     target_tokens = tokenizer.tokenize(examples.docs[i])[:args.max_target_length-2]
            # target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
            # target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            # padding_length = args.max_target_length - len(target_ids)
            # target_ids += [tokenizer.pad_token_id] * padding_length

            # features.append(
            #     InputFeatures(
            #         source_tokens,
            #         source_ids,
            #         source_mask,
            #         #  position_idx, 
            #         target_tokens,
            #         target_ids,
            #         #  examples.ast_diffs[i]
            #         source_ast_tokens
            #     )
            # )
            source_code_tokens = tokenizer.tokenize(examples.code_befores[i])
            source_ast_tokens = tokenizer.tokenize(examples.code_afters[i])

            source_tokens = source_code_tokens[:args.code_length-2]
            source_tokens = [tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            # position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

            source_ast_tokens = source_ast_tokens[:args.ast_length-1]
            source_ast_tokens = source_ast_tokens+[tokenizer.sep_token]
            source_tokens += [x for x in source_ast_tokens]
            source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_ast_tokens]
            
            source_mask = [1] * (len(source_tokens))
            padding_length = args.code_length + args.ast_length - len(source_ids)
            # position_idx += [tokenizer.pad_token_id] * padding_length
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask+=[0]*padding_length        

            #target
            if stage == 'test':
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(examples.docs[i])[:args.max_target_length-2]
            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length

            # features.append(
            #     InputFeatures(
            #         source_tokens,
            #         source_ids,
            #         source_mask,
            #         #  position_idx, 
            #         target_tokens,
            #         target_ids,
            #         source_ast_tokens
            #     )
            # )
        if args.model_type == 'codebert':
            source_code_tokens = tokenizer.tokenize(examples.code_befores[i])
            source_ast_tokens = tokenizer.tokenize(examples.code_afters[i])

            source_tokens = source_code_tokens[:args.code_length-2]
            source_tokens = [tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

            source_ast_tokens = source_ast_tokens[:args.ast_length-1]
            source_ast_tokens = source_ast_tokens+[tokenizer.sep_token]
            source_tokens += [x for x in source_ast_tokens]
            source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_ast_tokens]
            
            source_mask = [1] * (len(source_tokens))
            padding_length = args.code_length + args.ast_length - len(source_ids)
            position_idx += [tokenizer.pad_token_id] * padding_length
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask+=[0]*padding_length        

            #target
            if stage == 'test':
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(examples.docs[i])[:args.max_target_length-2]
            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length

            # features.append(
            #     InputFeatures(
            #         source_tokens,
            #         source_ids,
            #         source_mask,
            #         #  position_idx, 
            #         target_tokens,
            #         target_ids,
            #         # source_ast_tokens,
            #         [],
            #         []
            #     )
            # )

        if args.model_type == 'roberta':
            # source_code_tokens = tokenizer.tokenize(examples.code_befores[i])
            # source_ast_tokens = tokenizer.tokenize(examples.code_afters[i])

            # source_tokens = source_code_tokens[:args.code_length-2]
            # source_tokens = [tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
            # source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            # position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

            # source_ast_tokens = source_ast_tokens[:args.ast_length-1]
            # source_ast_tokens = source_ast_tokens+[tokenizer.sep_token]
            # source_tokens += [x for x in source_ast_tokens]
            # source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_ast_tokens]
            
            # source_mask = [1] * (len(source_tokens))
            # padding_length = args.code_length + args.ast_length - len(source_ids)
            # position_idx += [tokenizer.pad_token_id] * padding_length
            # source_ids += [tokenizer.pad_token_id] * padding_length
            # source_mask+=[0]*padding_length        

            # #target
            # if stage == 'test':
            #     target_tokens = tokenizer.tokenize("None")
            # else:
            #     target_tokens = tokenizer.tokenize(examples.docs[i])[:args.max_target_length-2]
            # target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
            # target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            # padding_length = args.max_target_length - len(target_ids)
            # target_ids += [tokenizer.pad_token_id] * padding_length

            # features.append(
            #     InputFeatures(
            #         source_tokens,
            #         source_ids,
            #         source_mask,
            #         #  position_idx, 
            #         target_tokens,
            #         target_ids,
            #         source_ast_tokens
            #     )
            # )

            source_code_tokens = tokenizer.tokenize(examples.code_diffs[i])
            ast_diffs = " ".join(examples.ast_diffs[i])
            source_ast_tokens = tokenizer.tokenize(ast_diffs)

            source_tokens = source_ast_tokens[:args.ast_length-2]
            source_tokens = [tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

            source_code_tokens = source_ast_tokens[:args.code_length-1]
            source_code_tokens = source_code_tokens+[tokenizer.sep_token]

            source_tokens += [x for x in source_code_tokens]
            source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_code_tokens]
            
            source_mask = [1] * (len(source_tokens))
            padding_length = args.code_length + args.ast_length - len(source_ids)
            # position_idx += [tokenizer.pad_token_id] * padding_length
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask+=[0]*padding_length 
            
            #target
            if stage == 'test':
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(examples.docs[i])[:args.max_target_length-2]
            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length

            # features.append(
            #     InputFeatures(
            #         source_tokens,
            #         source_ids,
            #         source_mask,
            #         #  position_idx, 
            #         target_tokens,
            #         target_ids,
            #         #  examples.ast_diffs[i]
            #         source_ast_tokens
            #     )
            # )

        if args.model_type == 'unixcoder':
            #for unixcoder
            # source_code_tokens = tokenizer.tokenize(examples.code_befores[i])
            # source_ast_tokens = tokenizer.tokenize(examples.code_afters[i])

            # source_tokens = source_code_tokens[:args.code_length-4]
            # source_tokens = [tokenizer.cls_token, "<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
            # source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

            # source_ast_tokens = source_ast_tokens[:args.ast_length-2]
            # source_ast_tokens = source_ast_tokens+[tokenizer.sep_token]
            # source_tokens += [x for x in source_ast_tokens]
            # source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_ast_tokens]

            # for diff

            source_code_tokens = []
            for source_code in examples.code_diffs[i]:
                source_code_tokens += tokenizer.tokenize(source_code)
            # source_ast_tokens = []
            # for ast_diff in examples.ast_diffs[i]:
            #     source_ast_tokens += tokenizer.tokenize(ast_diff)

            # if i == 0:
            #     logger.info('---example----')
            #     logger.info(f"code_diffs:{examples.code_diffs[i]}")
            #     logger.info(f"{' '.join(examples.code_diffs[i])}")
            #     logger.info(f"ast_diffs:{examples.ast_diffs[i]}")
            #     logger.info(f"token_code:{source_code_tokens}")
            #     logger.info(f"token_ast:{ast_diffs}")
            #     logger.info(f"token_after:{source_ast_tokens}")


            source_tokens = source_code_tokens[:args.code_length-5]
            source_tokens = [tokenizer.cls_token, "<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            # source_ast_tokens = source_ast_tokens[:args.ast_length-2]
            # source_ast_tokens = source_ast_tokens+[tokenizer.sep_token]
            # source_tokens += [x for x in source_ast_tokens]
            # source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_ast_tokens]

            #all
            source_mask = [1] * (len(source_tokens))
            # padding_length = args.code_length + args.ast_length - len(source_ids)
            padding_length = args.code_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask+=[0]*padding_length    

            #for diffrent level
            line_level_code = [0,0,0,0] + examples.lines_level[i][:args.code_length-4]
            token_level_code = [0,0,0,0] + examples.tokens_level[i][:args.code_length-4]
            padding_length_for_level = args.code_length - len(line_level_code)
            line_level_code += [0] * padding_length_for_level
            token_level_code += [0] * padding_length_for_level

            # logger.info('-----size of source------')
            # logger.info(len(source_code_tokens), len(line_level_code),len(token_level_code))
            # logger.info('\n\n')


            # for old/new ast 
            old_ast = [tokenizer.cls_token, "<encoder-decoder>",tokenizer.sep_token,"<mask0>"] + ast_changes.all_old_nodes[i][:args.code_length-5] + [tokenizer.sep_token]
            new_ast = [tokenizer.cls_token, "<encoder-decoder>",tokenizer.sep_token,"<mask0>"] + ast_changes.all_new_nodes[i][:args.code_length-5] + [tokenizer.sep_token]
            old_ast_maps =  [0,0,0,0] + ast_changes.all_old_maps[i][:args.code_length-4]
            new_ast_maps =  [0,0,0,0] + ast_changes.all_new_maps[i][:args.code_length-4]
            old_ast_ids = tokenizer.convert_tokens_to_ids(old_ast)
            new_ast_ids = tokenizer.convert_tokens_to_ids(new_ast)

            old_ast_mask = [1] * (len(old_ast))
            # padding_length = args.code_length + args.ast_length - len(old_ast_ids)
            padding_length = args.code_length - len(old_ast_ids)
            old_ast_ids += [tokenizer.pad_token_id] * padding_length
            old_ast_mask+=[0]*padding_length 

            new_ast_mask = [1] * (len(new_ast))
            padding_length = args.code_length - len(new_ast_ids)
            new_ast_ids += [tokenizer.pad_token_id] * padding_length
            new_ast_mask+=[0]*padding_length 

            old_ast_maps += [0] * (args.code_length - len(old_ast_maps))
            new_ast_maps += [0] * (args.code_length - len(new_ast_maps))

            # logger.info('--------------size of ast----------')
            # logger.info(f'{len(source_code_tokens)}, {len(old_ast_ids)}, {len(new_ast_ids)}, {len(old_ast_maps)}, {len(old_ast_mask)}')

            #target
            if stage == 'test':
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(examples.docs[i])[:args.max_target_length-2]
            target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length

            features.append(
                InputFeatures(
                    source_tokens,
                    source_ids,
                    source_mask,
                    target_tokens,
                    target_ids,
                    examples.labels[i],
                    # examples.lines_level[i],
                    # examples.tokens_level[i]
                    line_level_code,
                    token_level_code,
                    # for ast
                    old_ast_ids,
                    old_ast_mask,
                    new_ast_ids,
                    new_ast_mask,
                    old_ast_maps,
                    new_ast_maps
                )
            )

        if args.model_type == 't5':
            prefix = tokenizer.tokenize('summarize:')
            source_code_tokens = tokenizer.tokenize(examples.code_befores[i])
            source_ast_tokens = tokenizer.tokenize(examples.code_afters[i])

            source_tokens = prefix + source_code_tokens
            source_tokens = [tokenizer.cls_token]+source_tokens[:args.code_length-2]+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            source_ast_tokens = source_ast_tokens[:args.ast_length-1]
            source_ast_tokens = source_ast_tokens+[tokenizer.sep_token]
            source_tokens += [x for x in source_ast_tokens]
            source_ids += [tokenizer.convert_tokens_to_ids(x) for x in source_ast_tokens]
            
            source_mask = [1] * (len(source_tokens))
            padding_length = args.code_length + args.ast_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask+=[0]*padding_length  
            
            #target
            if stage == 'test':
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(examples.docs[i])[:args.max_target_length-2]
            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length

            # features.append(
            #     InputFeatures(
            #         source_tokens,
            #         source_ids,
            #         source_mask,
            #         #  position_idx, 
            #         target_tokens,
            #         target_ids,
            #         #  examples.ast_diffs[i]
            #         source_ast_tokens
            #     )
            # )
    return  features


