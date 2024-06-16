
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as f

from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import BartClassificationHead, shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqSequenceClassifierOutput

from tqdm import tqdm
import numpy as np
import logging

import enums
from .utils import inputs_to_cuda

logger = logging.getLogger(__name__)

class BartForClassificationAndGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig, mode=None):
        super(BartForClassificationAndGeneration, self).__init__(config)
        self.mode = None
        if mode:
            self.set_model_mode(mode)

        # classification head
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def set_model_mode(self, mode):
        assert mode in [enums.MODEL_MODE_GEN, enums.MODEL_MODE_CLS, enums.MODEL_MODE_SEARCH]
        self.mode = mode
        logging.info(f'BART mode switched to {mode}')

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            neg_nl_input_ids=None,
            neg_nl_attention_mask=None
    ):

        assert self.mode, 'It is required to specific a mode for BART before the model is passed through'
