import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mul_att(layers, x, y):
    querys, keys, vals, fuse = layers
    c = list()
    for i in range(len(querys)):
        q, k, v = querys[i](y), keys[i](x), vals[i](x)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        a = F.softmax(d, dim=-1)
        c_i = torch.matmul(a, v)
        c.append(c_i)
    x = torch.cat(c, dim=-1)
    return fuse(x)


class AttEncode(nn.Module):
    def __init__(self, en_embed_mat, pos_mat, head, stack):
        super(AttEncode, self).__init__()
        en_vocab_num, en_embed_len = en_embed_mat.size()
        self.en_embed = nn.Embedding(en_vocab_num, en_embed_len, _weight=en_embed_mat)
        self.pos = pos_mat
        self.querys, self.keys, self.vals = [[[nn.Linear(en_embed_len, 200)] * head] * stack] * 3
        self.fuses = [nn.Linear(200 * head, 200)] * stack
        self.lals = [nn.Sequential(nn.Linear(200, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 200))] * stack

    def forward(self, x):
        x = self.en_embed(x)
        x = x + self.pos
        for i in range(len(self.querys)):
            r = x
            layers = [self.querys[i], self.keys[i], self.vals[i], self.fuses[i]]
            x = mul_att(layers, x, x)
            x = F.layer_norm(x + r, x.size()[1:])
            r = x
            x = self.lals[i](x)
            x = F.layer_norm(x + r, x.size()[1:])
        return x


class AttDecode(nn.Module):
    def __init__(self, zh_embed_mat, pos_mat, head, stack):
        super(AttDecode, self).__init__()
        zh_vocab_num, zh_embed_len = zh_embed_mat.size()
        self.zh_embed = nn.Embedding(zh_vocab_num, zh_embed_len, _weight=zh_embed_mat)
        self.pos = pos_mat
        self.querys, self.keys, self.vals = [[[[nn.Linear(zh_embed_len, 200)] * head] * stack] * 2] * 3
        self.fuses = [[nn.Linear(200 * head, 200)] * stack] * 2
        self.lals = [nn.Sequential(nn.Linear(200, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 200))] * stack
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, zh_vocab_num))

    def forward(self, x, y):
        y = self.zh_embed(y)
        y = y + self.pos
        for i in range(len(self.querys)):
            r = y
            layers = [self.querys[0][i], self.keys[0][i], self.vals[0][i], self.fuses[0][i]]
            y = mul_att(layers, y, y)
            y = F.layer_norm(y + r, y.size()[1:])
            r = y
            layers = [self.querys[1][i], self.keys[1][i], self.vals[1][i], self.fuses[1][i]]
            y = mul_att(layers, y, x)
            y = F.layer_norm(y + r, y.size()[1:])
            r = y
            y = self.lals[i](y)
            y = F.layer_norm(y + r, y.size()[1:])
        return self.dl(y)
