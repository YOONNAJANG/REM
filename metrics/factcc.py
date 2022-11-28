# Copyright (c) 2020, Salesforce.com, Inc.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

# import json
# import logging
# import math
# import os
# import sys
# from io import open
# import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


#from transformers.modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, PreTrainedModel, prune_linear_layer, add_start_docstrings)
from transformers import BertPreTrainedModel, BertModel


class BertPointer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPointer, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # classifiers
        self.ext_start_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.ext_end_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.aug_start_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.aug_end_classifier = nn.Linear(config.hidden_size, 1, bias=False)

        self.label_classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        #self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,):
        # run through bert
        bert_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)

        # label classifier
        pooled_output = bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        label_logits = self.label_classifier(pooled_output)

        outputs = label_logits

        return outputs