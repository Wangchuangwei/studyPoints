from turtle import forward
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer,RobertaConfig, RobertaModel, T5Tokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from fastNLP import seq_len_to_mask

import logging
import os

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),     #for t5
                'unixcoder':(RobertaConfig, RobertaModel, RobertaTokenizer),
                'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer),
                'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),}

def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    logger.info(f"hidden_size:{config.hidden_size}")

    if args.model_type == 'codebert':
        encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq_codebert(encoder=encoder, decoder=decoder, config=config,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    elif args.model_type == 'codet5':
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        # decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
    
        model = Seq2Seq(model=model,config=config,tokenizer=tokenizer,args=args,
                    beam_size=args.beam_size,max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    elif args.model_type == 'unixcoder':
        # import！！！you must set is_decoder as True for generation
        config.is_decoder = True
        encoder = model_class.from_pretrained(args.model_name_or_path,config=config) 
        config.add_cross_attention=True
        decoder = model_class.from_pretrained(args.model_name_or_path,config=config) 
        # encoder.embeddings.word_embeddings = nn.Embedding(51419,768,padding_idx=1)
        # logger.info(f"mask:{tokenizer.mask_token_id}")

        # model = Seq2Seq_unixcoder(encoder=encoder,decoder=encoder,config=config,
        #             beam_size=args.beam_size,max_length=args.max_target_length,
        #             sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)

        model = Seq2Seq_pretrain(encoder=encoder,decoder=decoder,config=config,
                    beam_size=args.beam_size,max_length=args.max_target_length,
                    batch_size=args.train_batch_size, source_length=args.code_length,
                    sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)

    logger.info("Finish loading model [%s] from %s", args.model_type, args.model_name_or_path)
    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        # load_model_dir = os.path.join(args.checkpoint_root, 'unixbest', 'pytorch_model.bin') 
        load_model_dir = os.path.join(args.pre_train_output_root, 'cg', 'pytorch_model.bin') 
        model.load_state_dict(torch.load(load_model_dir), strict = False)

    return model, tokenizer

class ClassificationHead(nn.Module):
    """Head for token-level classification tasks."""
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 3)

    def forward(self, x):
        logger.info(f"size:{x.size()}")
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class Seq2Seq(nn.Module):

    def __init__(self, model, config, tokenizer, args,beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.args = args

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id  
    
    def get_t5_vec(self, source_ids,target_ids):
        # logger.info("get into1 t5_vec")
        # logger.info(f"eos_mask1_size:{source_ids}")
        # logger.info(f"eos_mask2_size:{target_ids}")
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        target_mask = target_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model(input_ids=source_ids, attention_mask=attention_mask,
                               labels=target_ids, decoder_attention_mask=target_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]  

        # logger.info(f"asdfsf{hidden_states}")
        return outputs, hidden_states 

    def forward_dp(self,source_ids, source_mask, target_ids):

        # logger.info(f"shape:{source_ids.shape[0]}")
        source_ids = source_ids.view(-1, self.args.code_length + self.args.ast_length)
        _, hidden_states = self.get_t5_vec(source_ids, source_mask)
        logits = self.classifier(hidden_states)
        # logger.info(f"size:{hidden_states.size()}")
        # logger.info(f"see:{hidden_states}")
        # logger.info(f"tokens:{self.tokenizer.convert_ids_to_tokens(source_mask)}")
        logger.info(f"source_mask:{source_mask}")
        logger.info(f"logits_be:{logits}")
        

        active_loss = source_ids.eq(self.tokenizer.mask_token_id).view(-1)
        shift_logits = logits[..., :-1, :].contiguous()    
        shift_labels = target_ids[..., 1:].contiguous()   

        logger.info(f"active_loss:{active_loss}")
        logger.info(f"view：{shift_logits.view(-1, shift_logits.size(-1))[active_loss]}")
        logger.info(f"target_ids：{shift_labels.view(-1)[active_loss]}")

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        # soft_input = torch.nn.Softmax(dim=0)
        # ites = soft_input(shift_logits.view(-1, shift_logits.size(-1))[active_loss])
        # logger.info(f"sorferf:{ites}")

        # loss = loss_fct(logits, target_ids)

        return loss, logits

    def forward_cg(self, source_ids, target_ids):
        source_ids = source_ids.view(-1, self.args.code_length + self.args.ast_length)
        _, hidden_states = self.get_t5_vec(source_ids,target_ids)
        x = torch.tanh(self.dense(hidden_states)).contiguous()
        logits = self.lm_head(x)

        active_loss = target_ids[..., 1:].ne(self.tokenizer.pad_token_id).view(-1)
        shift_logits = logits[..., :-1, :].contiguous()    
        shift_labels = target_ids[..., 1:].contiguous()   

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        return loss, loss* active_loss.sum() , active_loss.sum()

    def forward(self, source_ids, task, source_mask=None, target_ids=None):
        if task == 'dp':
            return self.forward_dp(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids)

        elif task == 'mass':
            outputs, _ = self.get_t5_vec(source_ids=source_ids, target_ids=source_mask)
            return outputs.loss


        elif task == 'cap':
            outputs, _ = self.get_t5_vec(source_ids=source_ids, target_ids=source_mask)
            return outputs.loss

        elif task == 'cg':
            if source_mask is not None:
                # return self.forward_cg(source_ids=source_ids, target_ids=source_mask)
                
                attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                target_mask_1 = source_mask.ne(self.tokenizer.pad_token_id)

                # codet5
                outputs = self.model(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_mask, decoder_attention_mask=target_mask_1, output_hidden_states=True)
                return outputs.loss

                # #codebert
                # return self.model(source_ids=source_ids,source_mask=attention_mask,target_ids=source_mask,target_mask=target_mask_1)

                

            else:
                attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                
                #codet5
                preds = self.model.generate(source_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        num_beams=self.args.beam_size,
                        early_stopping=True,
                        max_length=self.args.max_target_length)
                return preds
                
                # #codebert
                # return self.model(source_ids=source_ids,source_mask=attention_mask)
                

# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/model.py
class Seq2Seq_codebert(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq_codebert, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        outputs = outputs[0].permute([1, 0, 2]).contiguous()
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, outputs, tgt_mask=attn_mask,
                               memory_key_padding_mask=~source_mask)
            # memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = outputs[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=~context_mask)
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds

class Seq2Seq_unixcoder(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq_unixcoder, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id       
        
    def forward(self, source_ids, target_ids=None):   
        if target_ids is None:
            return self.generate(source_ids)
        
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        outputs = self.encoder(source_ids,attention_mask=mask,use_cache=True)  
        ids = torch.cat((source_ids,target_ids),-1)
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        mask = mask & ids[:,None,:].ne(1)

        out = self.decoder(target_ids,attention_mask=mask,past_key_values=outputs.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = loss,loss*active_loss.sum(),active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        outputs = self.encoder(source_ids,attention_mask=mask,use_cache=True)        
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in outputs.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break

                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   

class Seq2Seq_pretrain(nn.Module):       
    def __init__(self, encoder,decoder, config, beam_size=None, max_length=None, batch_size=None, source_length=None,sos_id=None, eos_id=None):
        super(Seq2Seq_pretrain, self).__init__()
        self.encoder = encoder

        self.encoder_old = encoder
        self.encoder_new = encoder

        self.decoder=decoder
        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        # self.classifier = nn.Linear(config.hidden_size, 3)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.source_length = source_length
        self.sos_id = sos_id
        self.eos_id = eos_id   
        self.pad_id = 1

        self.multihead_attn_line = nn.MultiheadAttention(config.hidden_size, 8)
        self.multihead_attn_token = nn.MultiheadAttention(config.hidden_size, 8)  
        self.layernorm_line =  nn.LayerNorm([batch_size, source_length, config.hidden_size])
        self.layernorm_token =  nn.LayerNorm([batch_size, source_length, config.hidden_size])

        self.multihead_attn_old = nn.MultiheadAttention(config.hidden_size, 8)
        self.multihead_attn_new = nn.MultiheadAttention(config.hidden_size, 8)  
        self.layernorm_old =  nn.LayerNorm([batch_size, source_length, config.hidden_size])
        self.layernorm_new =  nn.LayerNorm([batch_size, source_length, config.hidden_size])

        self.linear_layer = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.linear_layer.weight)

    def pad2max_len(self, input_tensor, max_len):
        pad_size = max_len - input_tensor.shape[-1]
        pad_tensor = torch.full([input_tensor.shape[0], input_tensor.shape[1], pad_size], self.pad_id,
                                device=input_tensor.device).long()
        return torch.cat([input_tensor, pad_tensor], dim=-1)
    
    def ranking_loss(self, cos_distance, bleu_distance):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        margin = 0.01
        ones = torch.ones(cos_distance.size(), device=cos_distance.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(cos_distance, cos_distance, ones)

        # candidate loss
        n = cos_distance.size(1)
        for i in range(1, n):
            pos_score = cos_distance[:, :-i]
            neg_score = cos_distance[:, i:]
            same_mask = (torch.abs(bleu_distance[:, :-i] - bleu_distance[:, i:]) > margin).float()
            ones = torch.ones(pos_score.size(), device=cos_distance.device)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')  # batch x i
            marginal_loss = loss_func(pos_score, neg_score, ones)
            if same_mask.sum() > 0:
                total_loss += (marginal_loss * same_mask).sum() / same_mask.sum()

        return total_loss

    def affine_transformation(self, input_features, padding_mask, axis=1):
        length = torch.sum(padding_mask, axis=1) - 1
        padding_mask = seq_len_to_mask(length, max_len=padding_mask.shape[-1])
        trans_tmp = F.relu(self.linear_layer(input_features))  # batch
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, axis=axis)
        return trans_emb * (1 / length.unsqueeze(-1))

    def form_ngram(self, input_tensor, n=2):
        """
        input_tensor: batch x sample_num x seq_len
        return: batch x seq_len-3 x 4
        """
        bsz, cand_num, seq_len = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)
        seq_len_clip = seq_len - n + 1
        input_tensor_repeated = input_tensor[:, :, None, :].repeat(1, 1, seq_len_clip, 1)
        help_matrix_1 = torch.triu(torch.ones(seq_len, seq_len))
        help_matrix_2 = torch.triu(torch.ones(seq_len, seq_len), diagonal=n)
        help_matrix = (help_matrix_1 - help_matrix_2)[:seq_len_clip].bool()[None, None, :, :]
        ret_tensor = torch.masked_select(input_tensor_repeated, help_matrix.to(input_tensor.device))
        return ret_tensor.view(bsz, cand_num, seq_len_clip, n)

    def torch_bleu(self, ref_tensor, sys_tensor, pad_id, n_gram=2):
        """
        Calculates n-gram precision with brevity penalty. contributed by jinulee-v

        ref_tensor: batch x seq_len1
        sys_tensor: batch x sample_num x seq_len2
        """
        # Determine batch size, sample count(=beam size), n-gram
        bsz, sample_num = sys_tensor.size(0), sys_tensor.size(1)
        n = min(min(n_gram, ref_tensor.size(-1)), sys_tensor.size(-1))

        # Generate masks
        ref_padding = (~(ref_tensor == pad_id)).float()
        ref_padding[:, 0] = 1
        ref_ngram_mask = torch.arange(0, ref_padding.size(1), device=ref_padding.device) * torch.ones_like(ref_padding)
        ref_ngram_mask = torch.where(
            ref_ngram_mask < (torch.sum(ref_padding, dim=-1, keepdims=True) - n + 1),
            ref_padding, torch.zeros_like(ref_padding)
        )[:, :ref_ngram_mask.size(-1) - n + 1]
        sys_padding = (~(sys_tensor == pad_id)).float()
        sys_padding[:, 0] = 1
        sys_ngram_mask = torch.arange(0, sys_padding.size(-1), device=sys_padding.device) * torch.ones_like(sys_padding)
        sys_ngram_mask = torch.where(
            sys_ngram_mask < (torch.sum(sys_padding, dim=-1, keepdims=True) - n + 1),
            sys_padding, torch.zeros_like(sys_padding)
        )[:, :, :sys_ngram_mask.size(-1) - n + 1]

        # Get n-grams
        ref_tensor = ref_tensor * ref_padding  # mask out paddings
        sys_tensor = sys_tensor * sys_padding
        ref_tensor = ref_tensor[:, None, :].repeat(1, sample_num, 1)  # readjust ref size to match sys
        input_tensor1_ngram = self.form_ngram(ref_tensor, n).float()
        input_tensor2_ngram = self.form_ngram(sys_tensor, n).float()  # batch x sample_num x seq_len-(n-1) x n

        # Calculate similarity matrix
        sim_matrix = (torch.norm(  # Calculate L2 norm to find if N-gram in `sys`` is present in `ref``
            input_tensor2_ngram.unsqueeze(3) - input_tensor1_ngram.unsqueeze(2),
            p=2, dim=-1
        ) == 0.0).to(torch.float)
        # print(sim_matrix.size(), sys_ngram_mask.size(), ref_ngram_mask.size())
        sim_matrix *= sys_ngram_mask.unsqueeze(3) * ref_ngram_mask.unsqueeze(1).unsqueeze(2)
        sim_matrix = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)

        # Brevity penalty
        ref_len = torch.sum(ref_padding, dim=-1, keepdims=True)
        sys_len = torch.sum(sys_padding, dim=-1)
        bp = torch.exp(1 - (ref_len / sys_len))
        bp = torch.where(ref_len >= sys_len, bp, torch.ones_like(bp))

        return sim_matrix / torch.sum(sys_ngram_mask, dim=-1) * bp  # batch x sample_num

    @torch.no_grad()
    def sample_from_model(self, src_inp, src_pad_mask):
        batch_size = src_inp.size(0)
        # candidate_id = self.decoder.generate(
        #     input_ids=src_inp,
        #     attention_mask=src_pad_mask,
        #     # past_key_values=src_inp
        #     # num_return_sequences=self.beam_size,
        #     # num_beam_groups=self.beam_size,
        #     # # diversity_penalty=self.diversity_pen,
        #     # diversity_penalty=1.0,
        #     # num_beams=self.beam_size,
        #     # max_length=self.max_length + 2,
        #     # min_length= 5 + 1,  # +1 from or
        #     # # min_length=self.args.min_length + 1,  # +1 from or
        #     # no_repeat_ngram_size=4,
        #     # # no_repeat_ngram_size=self.args.no_repeat_ngram,
        #     # length_penalty=2.0,
        #     # # length_penalty=self.args.length_pen,
        #     # early_stopping=True,
        #     # # early_stopping=self.args.early_stop,
        # )
        candidate_id = self.generate(src_inp)
        return candidate_id.view(batch_size, self.beam_size, -1)
        
    def forward(self, source_ids, task, target_ids=None, line_level_code=None, token_level_code=None, old_ast_ids=None, new_ast_ids=None, old_ast_maps=None, new_ast_maps=None):   
        if task == 'cg':
            if target_ids is None:
                return self.generate(source_ids)
            # logger.info(f"ids_shape:{source_ids.shape}")
            # logger.info(f'view:{source_ids}')
            # logger.info(f"line_shape:{line_level_code.shape}")
            # logger.info(f'view:{line_level_code}')
            # logger.info(f"token_shape:{token_level_code.shape}")
            # logger.info(f'view:{token_level_code}')
            
            mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
            encoder_outputs = self.encoder(source_ids,attention_mask=mask,use_cache=True)  
            outputs = encoder_outputs[0]
            # logger.info(f'mask_start:{mask.size()}')

            # for ast level action
            old_mask = old_ast_ids.ne(1)[:,None,:]*old_ast_ids.ne(1)[:,:,None]
            old_encoder_outputs = self.encoder_old(old_ast_ids,attention_mask=old_mask,use_cache=True)  
            outputs_old = old_encoder_outputs[0]

            new_mask = new_ast_ids.ne(1)[:,None,:]*new_ast_ids.ne(1)[:,:,None]
            new_encoder_outputs = self.encoder_new(new_ast_ids,attention_mask=new_mask,use_cache=True)  
            outputs_new = new_encoder_outputs[0]

            q_old = (old_ast_maps.ne(0)[:,:,None] * outputs_old[:,:self.source_length,:]).permute([1,0,2]).contiguous()
            q_new = (new_ast_maps.ne(0)[:,:,None] * outputs_new[:,:self.source_length,:]).permute([1,0,2]).contiguous()

            old_key_value = outputs_old.permute([1,0,2]).contiguous()
            new_key_value = outputs_new.permute([1,0,2]).contiguous()
            attn_output_old, _ = self.multihead_attn_old(q_old, old_key_value, old_key_value)
            attn_output_new, _ = self.multihead_attn_new(q_new, new_key_value, new_key_value)
            ast_h_final = self.layernorm_old(attn_output_old.permute([1,0,2]).contiguous()) + self.layernorm_new(attn_output_new.permute([1,0,2]).contiguous())

            #action deal h
            # logger.info(f"outputs:{outputs.size()}")
            # logger.info(f'view_outpit:{outputs}')

            #for line level action
            # logger.info(f'siez of lien:{line_level_code.ne(0)[:,:,None].size()}')
            # logger.info(f"line_deal:{line_level_code.ne(0)[:,:,None]}")
            # logger.info(f"vaild:{outputs[:,:self.source_length,:]}")

            q_line = (line_level_code.ne(0)[:,:,None] * outputs[:,:self.source_length,:]).permute([1,0,2]).contiguous()
            q_token = (token_level_code.ne(0)[:,:,None] * outputs[:,:self.source_length,:]).permute([1,0,2]).contiguous()

            # merge diffrent level action
            key_value = outputs.permute([1,0,2]).contiguous()
            attn_output_line, _ = self.multihead_attn_line(q_line, key_value, key_value)
            attn_output_token, _ = self.multihead_attn_token(q_token, key_value, key_value)

            # logger.info('----layernorm----')
            # logger.info(f"size_atten:{attn_output_line.size()}")
            # logger.info(f"size_atten:{attn_output_token.size()}")

            h_final = self.layernorm_line(attn_output_line.permute([1,0,2]).contiguous()) + self.layernorm_token(attn_output_token.permute([1,0,2]).contiguous())
            # h_final = self.layernorm_line(attn_output_line.permute([1,0,2]).contiguous())
            # h_final = self.layernorm_line(attn_output_token.permute([1,0,2]).contiguous())


            # logger.info(f"view_atten_line:{attn_output_line}")
            # logger.info(f"size_atten:{attn_output_line.size()}")
            # logger.info(f"view_atten_token:{attn_output_token}")
            # logger.info(f"size_atten:{attn_output_token.size()}")
            # logger.info(f"view_h_final:{h_final}")
            # logger.info(f"size_h_final:{h_final.size()}")

            final_output = torch.cat((outputs[:,:self.source_length,:]+ ast_h_final + h_final, outputs[:,self.source_length:,:]), 1)
            # final_output = torch.cat((ast_h_final + h_final, outputs[:,self.source_length:,:]), 1)

            # logger.info(f"view_final_output:{final_output}")
            # logger.info(f"size_final_output:{final_output.size()}")

            ids = torch.cat((source_ids,target_ids),-1)
            mask_1 = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
            mask_1 = mask_1 & ids[:,None,:].ne(1)
            # logger.info(f"mask_final:{mask_1.size()}")

            # logger.info('----check decoder----')

            # encoder_hidden_states=final_output,      ,past_key_values=encoder_outputs.past_key_values

            out = self.decoder(target_ids,encoder_hidden_states=final_output,attention_mask=mask_1,past_key_values=encoder_outputs.past_key_values).last_hidden_state

            ###### Contrastive loss ######
            # logger.info('----check decoder----')
            # logger.info(f"size_atten:{final_output.size()}")
            # logger.info(f"size_atten:{source_ids.size()}")
            # logger.info(f"size_atten:{out.size()}")
            cand_ids = self.sample_from_model(source_ids, mask_1)  # batch x beam_size x seq_len
            # prepare contrastive learning
            batch_size = final_output.size(0)
            samples_from_batch = target_ids[None, :, :].repeat(batch_size, 1, 1)
            cand_len = cand_ids.size(2)
            samples_len = samples_from_batch.size(2)
            if samples_len < cand_len:
                samples_from_batch = self.pad2max_len(samples_from_batch, cand_len)
            else:
                samples_from_batch = samples_from_batch[:, :, :cand_len]
            samples_all = torch.cat([cand_ids, samples_from_batch], dim=1)  # batch x total_sample_num x seq_len
            actual_distance = self.torch_bleu(target_ids, samples_all, 1, 2)  # batch x total_sample_num
            distance_mask = (actual_distance < 0.99)  # use to mask the gold
            actual_distance_masked = actual_distance * distance_mask.float()
            sample_num = min(7, actual_distance_masked.size(1) - 1)
            actual_distance, actual_indices = torch.sort(actual_distance_masked, dim=-1, descending=True)
            sampled_actual_distance = actual_distance[:, :sample_num]
            sampled_actual_indices = actual_indices[:, :sample_num]
            # concat itself
            self_indices = torch.arange(0, batch_size).reshape(batch_size, 1).to(
                sampled_actual_indices.device) + cand_ids.size(1)  # manually add gold
            sampled_indices = torch.cat([self_indices, sampled_actual_indices], dim=-1)

            self_distance = torch.full([batch_size, 1], 1.0, device=sampled_actual_distance.device)
            sampled_bleu_distance = torch.cat([self_distance, sampled_actual_distance], dim=-1)
            dummy = sampled_indices.unsqueeze(-1).repeat(1, 1, samples_all.size(2))
            sampled_input = torch.gather(samples_all, 1, dummy)  # batch x sample_num x seq_len

            # logger.info(f"sampled_input:{sampled_input.size()}")
            decoder_hidden_states = []
            # logger.info('----chsssssssssssssk cssssss----')
            for sample_idx in range(sampled_indices.size(-1)):
                sampled_input_dec = sampled_input[:, sample_idx, :]

                sample_pad_mask = ~(sampled_input_dec == self.pad_id)
                sample_pad_mask[:, 0] = 1
                # logger.info('----aaaaaaaaaaaaaaaaaaaaaa----')
                # logger.info(f"sampled_input_dec:{sampled_input_dec.size()}")
                # logger.info(f"sample_pad_mask:{sample_pad_mask.size()}")
                # logger.info(f"final_output:{final_output.size()}")
                # logger.info(f"mask:{mask.size()}")
                decoder_out = self.decoder(input_ids=sampled_input_dec, attention_mask=sample_pad_mask,
                                    encoder_hidden_states=final_output).last_hidden_state
                                    # encoder_attention_mask=mask)  # last layer
                # logger.info(f"decoder_out:{decoder_out.size()}")
                # decoder_feature = decoder_out[0]  # batch x tgt_len x hidden
                decoder_feature = decoder_out  # batch x tgt_len x hidden
                # logger.info(f"decoder_feature:{decoder_feature.size()}")
                decoder_feature = self.affine_transformation(decoder_feature, sample_pad_mask)  # batch x h
                decoder_hidden_states.append(decoder_feature.unsqueeze(1))

            # logger.info('----check cssssss----')
            # logger.info(f"final_output:{final_output.size()}")
            # logger.info(f"mask:{mask.size()}")
            # logger.info(f"source_ids.ne(1):{source_ids.ne(1).size()}")
            encoder_feature = self.affine_transformation(final_output, source_ids.ne(1))  # batch x h
            # logger.info(f"encoder_feature:{encoder_feature.size()}")

            decoder_feature = torch.cat(decoder_hidden_states, dim=1)  # batch x sample_num x h
            # logger.info('----check cssssssssssaaaaaaaaasss----')
            # logger.info(f"decoder_feature:{decoder_feature.size()}")

            cos_distance = torch.cosine_similarity(encoder_feature.unsqueeze(1), decoder_feature,
                                                dim=-1)  # batch x samle_num
            # logger.info('----check csssaaasss----')

            cl_loss = self.ranking_loss(cos_distance, sampled_bleu_distance)

            # logger.info('----check cl_loss----')
            # logger.info(f"cl_loss:{cl_loss}")


            return self.forward_cg(source_ids=source_ids, target_ids=target_ids, out=out, cl_loss=cl_loss)

    def forward_cg(self, source_ids, target_ids, out, cl_loss):
        logits = self.lm_head(out)
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = logits[..., :-1, :].contiguous()    
        shift_labels = target_ids[..., 1:].contiguous()   

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        return loss + cl_loss, loss* active_loss.sum() + cl_loss , active_loss.sum() + cl_loss

    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        outputs = self.encoder(source_ids,attention_mask=mask,use_cache=True)        
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in outputs.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break

                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   


class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
