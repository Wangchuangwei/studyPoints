import torch
import torch.utils.data
from transformers import AdamW, get_linear_schedule_with_warmup, T5Config, T5ForConditionalGeneration, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.nn as nn

import torch.distributed as dist

import os
import logging
from typing import Union, Tuple
from tqdm import tqdm, trange
import numpy as np
import random

import enums
from utils import bleu, smooth_bleu
from utils.general import count_params, human_format, layer_wise_parameters
from utils.model import Seq2Seq, build_or_load_gen_model
from utils.eval.meteor import Meteor
from utils.eval.rouge import Rouge
from data.dataset import init_dataset, TextDataset, init_ast_dataset
from data.data_utils import InputFeatures, convert_examples_to_features

logger = logging.getLogger(__name__)

def train(args, tokenizer=None,
        trained_model: Union[Seq2Seq, str] = None):
    """
    Fine-tuning from given pre-trained model 

    """
    task = args.task.lower()
    assert task in enums.ALL_DOWNSTREAM_TASKS, f'Downstream task {task} is not supported.'

    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading datasets')
    datasets = dict()
    ast_datasets = dict()
    # splits = ['test'] if only_test else ['train', 'valid', 'test']
    # splits = ['test']
    splits = ['train', 'val','test']
    # splits = ['train']
    for split in splits:
        datasets[split] = init_dataset(args=args,
                                       mode=enums.TRAINING_MODE_FINE_TUNE,
                                       task=enums.TASK_COMMNET_GENERATION,
                                       stage=split)
        ast_datasets[split] = init_ast_dataset(args=args,
                                       mode=enums.TRAINING_MODE_FINE_TUNE,
                                       task=enums.TASK_COMMNET_GENERATION,
                                       stage=split)
        logger.info(f'The size of after process {split} set: {datasets[split].size} ast: {ast_datasets[split].size}')
        # logger.info(f'The size of after process {split} set: {len(datasets[split])}')

    logger.info('Datasets loaded successfully')
    # return
    
    # --------------------------------------------------
    # model
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_model is  not None or args.trained_model is not None:
        if isinstance(trained_model, Seq2Seq):
        # if trained_model is not None:
            logger.info('Model is passed through parameter')
            model = trained_model
            tokenizer = tokenizer
        else:
            model, tokenizer = build_or_load_gen_model(args)
            logger.info('Loading the model from file')
            model_prefix = 'pytorch_model.bin'
            output_dir = os.path.join(args.pre_train_output_root, 'cg', model_prefix)  
            # output_dir = os.path.join(args.checkpoint_root, 'unixbest',model_prefix)
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir), strict = False)                                      
    else:
        logger.info('Building the model')
        model, tokenizer = build_or_load_gen_model(args)

    # log model statistics
    logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    logger.info(f'random_seed:{args.random_seed}')
    logger.info(f"learning_rate:{args.learning_rate}")
    logger.info(f"batch_size:{args.train_batch_size}")
    logger.info(f'code_length:{args.code_length}')
    logger.info(f'ast_length:{args.ast_length}')
    logger.info('Model built successfully')
    # log model statistics
    # logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    # table = layer_wise_parameters(model)
    # logger.debug('Layer-wised trainable parameters:\n{}'.format(table))

    # --------------------------------------------------
    # train
    # --------------------------------------------------
    
    # Setup CUDA, GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    if torch.cuda.device_count() > 1:
        # multi-gpu training  
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model = torch.nn.DataParallel(model)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

        # model = nn.parallel.DistributedDataParallel(
        #     model,
        #     device_ids=[0, 1],
        #     output_device=0
        # )


    model.to(device)

    if args.do_train:
        logger.info('-' * 100)
        logger.info('Start training')
        #Prepare training data loader
        train_length = datasets["train"].size - datasets["train"].size % args.train_batch_size
        train_features = convert_examples_to_features(datasets["train"], tokenizer,args,size=train_length, stage='train', ast_changes=ast_datasets['train'])
        # train_features = convert_examples_to_features(datasets["train"], tokenizer,args,size=108, stage='train', ast_changes=ast_datasets['train'])
        # train_features = convert_examples_to_features(datasets["train"], tokenizer,args,size=100, stage='train', ast_changes=ast_datasets['train'])
        # train_features = convert_examples_to_features(datasets["train"], tokenizer,args,size=len(datasets["train"]), stage='train')
        # train_features = convert_examples_to_features(datasets["train"], tokenizer,args,size=504, stage='train')
        train_data = TextDataset(train_features,args,tokenizer=tokenizer)
        train_data.set_task(task)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps,num_workers=8, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # optimizer = nn.DataParallel(optimizer, device_ids=['0,1'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(datasets["train"]))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        
        patience, best_bleu, dev_dataset = 0, 0, {}
        best_f1=0
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss, global_step = 0, 0, 0, 0

            model.train()
            # logger.info()
            for step, batch in enumerate(bar):
                # logger.info(batch,'batch')
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids, labels, line_level_code, token_level_code, old_ast_ids, new_ast_ids, old_ast_maps, new_ast_maps = batch
                # source_ids, target_ids, line_level_code, token_level_code = batch
                # logger.info(f"pro: {source_ids.shape},see:{old_ast_maps.shape}, ast: {old_ast_ids.shape}")

                if args.model_type == 'codebert':
                #codebert
                    attention_mask = source_ids.ne(tokenizer.pad_token_id)
                    target_mask = target_ids.ne(tokenizer.pad_token_id)
                    loss,_,_ = model(source_ids=source_ids,source_mask=attention_mask,target_ids=target_ids,target_mask=target_mask)    
                # 
                elif args.model_type == 'codet5':
                #codet5
                    loss = model(source_ids=source_ids,task="cg",source_mask=target_ids)
                elif args.model_type == 'unixcoder':
                    if args.task == 'cg':
                        loss,_,_ = model(source_ids=source_ids,task=args.task,target_ids=target_ids, line_level_code=line_level_code, token_level_code=token_level_code, old_ast_ids=old_ast_ids, new_ast_ids=new_ast_ids, old_ast_maps=old_ast_maps, new_ast_maps=new_ast_maps)
                    else:
                        loss,logits = model(source_ids=source_ids,task=args.task,target_ids=target_ids, labels=labels,line_level_code=line_level_code, token_level_code=token_level_code, old_ast_ids=old_ast_ids, new_ast_ids=new_ast_ids, old_ast_maps=old_ast_maps, new_ast_maps=new_ast_maps)
                elif args.model_type == 't5':
                    attention_mask = source_ids.ne(tokenizer.pad_token_id)
                    target_mask = target_ids.ne(tokenizer.pad_token_id)
                    outputs= model(input_ids=source_ids,attention_mask=attention_mask,labels=target_ids,decoder_attention_mask=target_mask)    
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                # break

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            
            if args.do_eval:
                #Eval model with dev dataset                  
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    val_length = datasets["val"].size - datasets["val"].size % args.train_batch_size
                    eval_features = convert_examples_to_features(datasets["val"], tokenizer, args,size=val_length, stage='dev',ast_changes=ast_datasets['val'])
                    # eval_features = convert_examples_to_features(datasets["val"], tokenizer, args,size=len(datasets["val"]), stage='dev')
                    eval_data = TextDataset(eval_features,args,tokenizer=tokenizer)
                    eval_data.set_task(task)
                    dev_dataset['dev_loss']=datasets["val"],eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=8)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(datasets["val"]))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,batch_num = 0,0
                logits = []
                y_trues = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)               
                    source_ids, target_ids, labels,line_level_code, token_level_code, old_ast_ids, new_ast_ids, old_ast_maps, new_ast_maps = batch
                    with torch.no_grad():
                        if args.model_type == 'codebert':
                        #codebert
                            attention_mask = source_ids.ne(tokenizer.pad_token_id)
                            target_mask = target_ids.ne(tokenizer.pad_token_id)
                            _,loss,num = model(source_ids=source_ids,source_mask=attention_mask,target_ids=target_ids,target_mask=target_mask)
                        elif args.model_type == 'codet5':
                        #codet5
                            loss = model(source_ids=source_ids,task='cg',source_mask=target_ids)
                            num = 1
                        elif args.model_type == 'unixcoder':
                            # _,loss,num = model(source_ids=source_ids,target_ids=target_ids) 
                            #mine pretrain
                            if task == 'cg':
                                _,loss,num = model(source_ids=source_ids,task=args.task,target_ids=target_ids, line_level_code=line_level_code, token_level_code=token_level_code, old_ast_ids=old_ast_ids, new_ast_ids=new_ast_ids, old_ast_maps=old_ast_maps, new_ast_maps=new_ast_maps) 
                            else:
                                loss,logit = model(source_ids=source_ids,task=args.task,target_ids=target_ids, labels=labels,line_level_code=line_level_code, token_level_code=token_level_code, old_ast_ids=old_ast_ids, new_ast_ids=new_ast_ids, old_ast_maps=old_ast_maps, new_ast_maps=new_ast_maps) 
                        elif args.model_type == 't5':
                            attention_mask = source_ids.ne(tokenizer.pad_token_id)
                            target_mask = target_ids.ne(tokenizer.pad_token_id)
                            outputs= model(input_ids=source_ids,attention_mask=attention_mask,labels=target_ids,decoder_attention_mask=target_mask)    
                            loss = outputs.loss
                        
                    eval_loss += loss.item()
                    if args.task == 'cg':
                        batch_num += num.sum().item() # num.sum().item()  1
                    else:
                        logits.append(logit.cpu().numpy())
                        y_trues.append(labels.cpu().numpy())
                    # tokens_num += num.sum().item()

                #Pring loss of dev dataset  
                if args.task == 'cg':
                    eval_loss = eval_loss / batch_num
                    eval_ppl = round(np.exp(eval_loss), 5)
                    model.train()
                    result = {'epoch': epoch,
                            'eval_ppl': eval_ppl,
                            'global_step': global_step+1}
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                    logger.info("  "+"*"*20)   

                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()
                    
                    #Calculate bleu  
                    if 'dev_bleu' in dev_dataset:
                        eval_examples,eval_data=dev_dataset['dev_bleu']
                    else:
                        eval_examples = datasets["val"]
                        # eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,size=val_length,stage='test', ast_changes=ast_datasets['val'])
                        # eval_features = convert_examples_to_features(eval_examples, tokenizer, args,size=len(datasets["val"]),stage='test')
                        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long) 
                        eval_data = TensorDataset(all_source_ids)   
                        dev_dataset['dev_bleu'] = eval_examples,eval_data

                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval() 

                    p = []
                    for batch in eval_dataloader:
                        batch = tuple(t.to(device) for t in batch)
                        source_ids = batch[0]                  
                        with torch.no_grad():
                            if args.model_type == 'codebert':
                                attention_mask = source_ids.ne(tokenizer.pad_token_id)
                                preds = model(source_ids=source_ids,source_mask=attention_mask)   
                                for pred in preds:
                                    t=pred[0].cpu().numpy()
                                    t=list(t)
                                    if 0 in t:
                                        t=t[:t.index(0)]
                                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                                    p.append(text)
                            elif args.model_type == 'codet5':
                                preds = model(source_ids, task='cg') 
                                # convert ids to text
                                for pred in preds:
                                    t = pred.cpu().numpy()
                                    t = list(t)
                                    if 0 in t:
                                        t = t[t.index(0)+1:]
                                    text = tokenizer.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                                    p.append(text)
                            elif args.model_type == 'unixcoder':
                                # preds = model(source_ids)  
                                #mine pretrain
                                preds = model(source_ids,task='cg')
                                for pred in preds:
                                    t = pred[0].cpu().numpy()
                                    t = list(t)
                                    if 0 in t:
                                        t = t[:t.index(0)]
                                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                                    p.append(text)
                            elif args.model_type == 't5':
                                attention_mask = source_ids.ne(tokenizer.pad_token_id)
                                preds = model.generate(source_ids, 
                                        attention_mask=attention_mask,
                                        use_cache=True,
                                        num_beams=args.beam_size,
                                        early_stopping=True,
                                        max_length=args.max_target_length) 
                                # convert ids to text
                                for pred in preds:
                                    t = pred.cpu().numpy()
                                    t = list(t)
                                    if 0 in t:
                                        t = t[t.index(0)+1:]
                                    text = tokenizer.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                                    p.append(text)
                    
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()
                    
                    # model.train()
                    predictions = []
                    count = 0
                    with open(args.model_root+"/dev.output",'w') as f, open(args.model_root+"/dev.gold",'w') as f1:
                        for ref,gold in zip(p,eval_examples.docs):  
                            predictions.append(str(count)+'\t'+ref)
                            f.write(str(count)+'\t'+ref.strip()+'\n')
                            f1.write(str(count)+'\t'+gold.strip()+'\n')    
                            count += 1

                    (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, os.path.join(args.model_root, "dev.gold")) 
                    dev_bleu=round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                    logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                    logger.info("  "+"*"*20)    
                    if dev_bleu > best_bleu:
                        logger.info("  Best bleu:%s",dev_bleu)
                        logger.info("  "+"*"*20)
                        best_bleu = dev_bleu
                        # Save best checkpoint for best bleu
                        output_dir = args.checkpoint_root
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        
                        ##codet5
                        # output_model_file = os.path.join(output_dir, "codet5","pytorch_model.bin")

                        # #t5
                        # output_model_file = os.path.join(output_dir, "t5","pytorch_model.bin")

                        # #codebert
                        # output_model_file = os.path.join(output_dir, 'codebert',"pytorch_model.bin")

                        # #unixcoder
                        # output_model_file = os.path.join(output_dir, 'unixcoder',"pytorch_model.bin")

                        # #mine
                        # output_model_file = os.path.join(output_dir, 'unixbest',"pytorch_model.bin")

                        # #minepretrain
                        # output_model_file = os.path.join(output_dir, 'minepretrain',"pytorch_model.bin")

                        # #for study 2 + middle
                        # output_model_file = os.path.join(output_dir, 'study2', 'middle',"pytorch_model.bin")
                        
                        # #for study 2 + large
                        output_model_file = os.path.join(output_dir, 'study2', 'large',"pytorch_model.bin")

                        torch.save(model_to_save.state_dict(), output_model_file)
                        patience =0
                    else:
                        patience +=1
                        if patience ==2:
                            break
                else:
                    logits=np.concatenate(logits,0)
                    y_trues=np.concatenate(y_trues,0)
                    best_threshold=0.5
                    best_f1=0

                    y_preds=logits[:,1]>best_threshold
                    from sklearn.metrics import recall_score
                    recall=recall_score(y_trues, y_preds)
                    from sklearn.metrics import precision_score
                    precision=precision_score(y_trues, y_preds)   
                    from sklearn.metrics import f1_score
                    f1=f1_score(y_trues, y_preds)             
                    result = {
                        "eval_recall": float(recall),
                        "eval_precision": float(precision),
                        "eval_f1": float(f1),
                        "eval_threshold":best_threshold,
                        
                    }
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(round(result[key],4)))
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.checkpoint_root, 'study3','{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

            
    if args.do_test:
        checkpoint_prefix = 'pytorch_model.bin'

        # #codet5
        # output_dir = os.path.join(args.checkpoint_root, "codet5","checkpoint_prefix)  

        # #t5
        # output_dir = os.path.join(args.checkpoint_root, "t5",checkpoint_prefix)  

        # #codebert
        # output_dir = os.path.join(args.checkpoint_root, 'codebert', checkpoint_prefix) 

        # #unixcoder
        # output_dir = os.path.join(args.checkpoint_root, 'unixcoder', checkpoint_prefix) 

        # # #minebert
        # output_dir = os.path.join(args.checkpoint_root, 'unixbest', checkpoint_prefix) 

        # # minepretrain
        # output_dir = os.path.join(args.checkpoint_root, 'minepretrain', checkpoint_prefix) 

        # #for study 2 + middle
        # output_dir = os.path.join(args.checkpoint_root, 'study2', 'middle',"pytorch_model.bin")
        
        # #for study 2 + large 
        output_dir = os.path.join(args.checkpoint_root, 'study2', 'large',"pytorch_model.bin")

        if args.task != 'cg':
            output_dir = os.path.join(args.checkpoint_root, 'study3', 'checkpoint-best-f1',"pytorch_model.bin")
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))  

        eval_examples = datasets['test']
        # logger.info(f'The size of {split} set: {len(eval_examples)}')
        test_length = datasets["test"].size - datasets["test"].size % args.train_batch_size
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,size=test_length,stage='test', ast_changes=ast_datasets['test'])
        # eval_features = convert_examples_to_features(eval_examples, tokenizer, args,size=len(datasets["test"]),stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids)   

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running Test *****")
        model.eval() 
        p=[]
        logits=[]  
        y_trues=[]
        eval_loss = 0.0
        nb_eval_steps = 0
        best_threshold=0.5
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids, target_ids, labels,line_level_code, token_level_code, old_ast_ids, new_ast_ids, old_ast_maps, new_ast_maps = batch                
            with torch.no_grad():
                if args.task == 'cg':
                    # preds = model(source_ids,task='cg') 

                    ##codet5
                    # for pred in preds:
                    #     t = pred.cpu().numpy()
                    #     t = list(t)
                    #     if 0 in t:
                    #         t = t[t.index(0)+1:]
                    #     text = tokenizer.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    #     p.append(text)

                    #codebert
                    # for pred in preds:
                    #     t=pred[0].cpu().numpy()
                    #     t=list(t)
                    #     if 0 in t:
                    #         t=t[:t.index(0)]
                    #     text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    #     p.append(text)
                    if args.model_type == 'codebert':
                        attention_mask = source_ids.ne(tokenizer.pad_token_id)
                        preds = model(source_ids=source_ids,source_mask=attention_mask)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                    elif args.model_type == 'codet5':
                        preds = model(source_ids, task='cg') 
                        # convert ids to text
                        for pred in preds:
                            t = pred.cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[t.index(0)+1:]
                            text = tokenizer.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                            p.append(text)
                    elif args.model_type == 'unixcoder':
                        # preds = model(source_ids) 
                        #mine pretrain
                        preds = model(source_ids,task='cg')
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                    elif args.model_type == 't5':
                        attention_mask = source_ids.ne(tokenizer.pad_token_id)
                        preds = model.generate(source_ids, 
                                    attention_mask=attention_mask,
                                    use_cache=True,
                                    num_beams=args.beam_size,
                                    early_stopping=True,
                                    max_length=args.max_target_length) 
                        # convert ids to text
                        for pred in preds:
                            t = pred.cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[t.index(0)+1:]
                            text = tokenizer.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                            p.append(text)
                else:
                    lm_loss,logit = model(source_ids=source_ids,task=args.task,target_ids=target_ids, labels=labels,line_level_code=line_level_code, token_level_code=token_level_code, old_ast_ids=old_ast_ids, new_ast_ids=new_ast_ids, old_ast_maps=old_ast_maps, new_ast_maps=new_ast_maps) 
                    eval_loss += lm_loss.mean().item()
                    logits.append(logit.cpu().numpy())
                    y_trues.append(labels.cpu().numpy())
            nb_eval_steps += 1
        # model.train()
        if args.task == 'cg':
            predictions=[]
            h_dict = dict()  #idx: prediction
            r_dict = dict()  #idx: ground truth
            count = 0
            with open(args.model_root+"/test.output",'w') as f, open(args.model_root+"/test.gold",'w') as f1:
                for ref,gold in zip(p,eval_examples.docs):
                    # predictions.append(str(gold.idx)+'\t'+ref)
                    # f.write(str(gold.idx)+'\t'+ref+'\n')
                    # f1.write(str(gold.idx)+'\t'+gold.target+'\n')    
                    predictions.append(str(count)+'\t'+ref)
                    f.write(str(count)+'\t'+ref.strip()+'\n')
                    f1.write(str(count)+'\t'+gold.strip()+'\n')  

                    h_dict[count] = [ref.strip()]
                    r_dict[count] = [gold.strip()]

                    count += 1 

            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, os.path.join(args.model_root, "test.gold")) 
            dev_bleu=round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  "+"*"*20)    



            meteor_calculator = Meteor()
            meteor, _ = meteor_calculator.compute_score(r_dict, h_dict)
            logger.info("  %s = %s "%("meteor",str(meteor * 100)))
            logger.info("  "+"*"*20) 
            
            rouge_calculator = Rouge()
            rouge_l, ind_rouge = rouge_calculator.compute_score(r_dict, h_dict)
            logger.info("  %s = %s "%("rouge_l",str(rouge_l * 100)))
            logger.info("  "+"*"*20) 

            logger.info(f"show nei rong:{h_dict[2]}")
            logger.info(f"show nei rong:{r_dict[2]}")
        else:
            #output result
            logits=np.concatenate(logits,0)
            y_preds=logits[:,1]>best_threshold
            # with open(os.path.join(args.preds_root,"predictions.txt"),'w') as f:
            #     for example,pred in zip(eval_dataset.examples,y_preds):
            #         if pred:
            #             f.write(example.url1+'\t'+example.url2+'\t'+'1'+'\n')
            #         else:
            #             f.write(example.url1+'\t'+example.url2+'\t'+'0'+'\n')
            from sklearn.metrics import recall_score
            recall=recall_score(y_trues, y_preds)
            from sklearn.metrics import precision_score
            precision=precision_score(y_trues, y_preds)   
            from sklearn.metrics import f1_score
            f1=f1_score(y_trues, y_preds)             
            result = {
                "eval_recall": float(recall),
                "eval_precision": float(precision),
                "eval_f1": float(f1),
                "eval_threshold":best_threshold,
                
            }
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],4)))
    





