#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import multiprocessing
import os
import torch
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB["bert.base"] = (d2l.DATA_URL + 'bert.base.torch.zip', 
                             "225d66f04cae318b841a13d32af3acc165f253ac")

d2l.DATA_HUB["bert.small"] = (d2l.DATA_URL + "bert.small.torch.zip",
                              "c72329e68a732bef0452e4b96a1c341c8910f81f")


def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):

    data_dir = d2l.download_extract(pretrained_model)
    # 定义空词表以加载预定义词表
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))

    vocab.token_to_idx = {
        token: idx for idx, token in enumerate(vocab.idx_to_token)
    }
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256], ffn_num_input=256, 
                         ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads, num_layers=num_layers,
                         dropout=dropout, max_len=max_len, key_size=256, query_size=256,value_size=256,
                         hid_in_features=256, mlm_in_features=256, nsp_in_features=256)

    # 加载BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir, "pretrained.params")))
    return bert, vocab


class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_permise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences]) for sentences in dataset[:2]])]
    
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments, self.valid_lens) = self._preprogress(
            all_permise_hypothesis_tokens
        )
        print("read " + str(len(self.all_token_ids)) + " examples")

    def _preprogress(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, _, __ in out
        ]

        all_segments = [segments for _, segments, __ in out]
        valid_lens = [valid_len for _, __, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long), 
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab["<pad>"]] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 为BERT 输入中的'<cls>', '<sep>' 和'<sep>' 词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len-3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()


    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], 
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)


class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256,3)

if __name__ == "__main__":
    devices = "cuda" if torch.cuda.is_available() else "cpu"
    bert, vocab = load_pretrained_model(
        "bert.small", num_hiddens=256, ffn_num_hiddens=512, num_heads=4, num_layers=2, 
        dropout=0.1, max_len=512, devices=devices
    )

    batch_size, max_len, num_workers = 24, 128, d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
    train_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers=num_workers)

