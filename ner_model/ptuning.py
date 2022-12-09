import torch
import torch.nn as nn
import json
from os.path import join


def init_vocab(args):
    global shared_vocab, lama_vocab
    shared_vocab = json.load(open(join(args.data_dir, '29k-vocab.json')))
    lama_vocab = json.load(open(join(args.data_dir, '34k-vocab.json')))


def token_wrapper(args, token):
    if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
        return 'Ġ' + token
    else:
        return token


def get_vocab(model_name, strategy):
    if strategy == 'shared':
        if 'gpt' in model_name:
            return shared_vocab['gpt2-xl']
        elif 'roberta' in model_name or 'megatron' in model_name:
            return shared_vocab['roberta-large']
        else:
            assert model_name in shared_vocab
            return shared_vocab[model_name]
    elif strategy == 'lama':
        if 'gpt' in model_name:
            return lama_vocab['gpt2-xl']
        elif 'roberta' in model_name or 'megatron' in model_name:
            return lama_vocab['roberta-large']
        else:
            assert model_name in lama_vocab
            return lama_vocab[model_name]


def get_vocab_by_strategy(args, tokenizer):
    if args.vocab_strategy == 'original':
        return tokenizer.get_vocab()
    else:
        return get_vocab(args.model_name, args.vocab_strategy)


def get_embedding_layer(args, model):
    # if 'BART' in args.model_name:
    embeddings = model.model.get_input_embeddings()
    # elif 'T5' in args.model_name:
    #     embeddings = model.get_input_embeddings()
    # else:
    #     raise NotImplementedError()
    return embeddings

class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)
        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds





# class PromptEncoder():
#     def __init__(self):
#