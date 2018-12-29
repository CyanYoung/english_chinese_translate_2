import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Att(nn.Module):
    def __init__(self, en_embed_mat, zh_embed_mat, stack):
        super(Att, self).__init__()
        self.en_vocab_num, self.en_embed_len = en_embed_mat.size()
        self.zh_vocab_num, self.zh_embed_len = zh_embed_mat.size()
        self.en_embed = nn.Embedding(self.en_vocab_num, self.en_embed_len, _weight=en_embed_mat)
        self.zh_embed = nn.Embedding(self.zh_vocab_num, self.zh_embed_len, _weight=zh_embed_mat)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, self.zh_vocab_num))

    def forward(self, x, y):
        x = self.en_embed(x)
        y = self.zh_embed(y)
        h1, h1_n = self.encode(x)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v)
        s2 = torch.cat((h2, c), dim=-1)
        return self.dl(s2)


class AttEncode(nn.Module):
    def __init__(self, en_embed_mat):
        super(AttEncode, self).__init__()
        self.en_vocab_num, self.en_embed_len = en_embed_mat.size()
        self.en_embed = nn.Embedding(self.en_vocab_num, self.en_embed_len)
        self.encode = nn.GRU(self.en_embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.en_embed(x)
        h1, h1_n = self.encode(x)
        return h1


class AttDecode(nn.Module):
    def __init__(self, zh_embed_mat):
        super(AttDecode, self).__init__()
        self.zh_vocab_num, self.zh_embed_len = zh_embed_mat.size()
        self.zh_embed = nn.Embedding(self.zh_vocab_num, self.zh_embed_len)
        self.decode = nn.GRU(self.zh_embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, self.zh_vocab_num))

    def forward(self, y, h1):
        y = self.zh_embed(y)
        h1_n = torch.unsqueeze(h1[:, -1, :], dim=0)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v)
        s2 = torch.cat((h2, c), dim=-1)
        return self.dl(s2)


class AttPlot(nn.Module):
    def __init__(self, en_embed_mat, zh_embed_mat):
        super(AttPlot, self).__init__()
        self.en_vocab_num, self.en_embed_len = en_embed_mat.size()
        self.zh_vocab_num, self.zh_embed_len = zh_embed_mat.size()
        self.en_embed = nn.Embedding(self.en_vocab_num, self.en_embed_len)
        self.zh_embed = nn.Embedding(self.zh_vocab_num, self.zh_embed_len)
        self.encode = nn.GRU(self.en_embed_len, 200, batch_first=True)
        self.decode = nn.GRU(self.zh_embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3

    def forward(self, x, y):
        x = self.en_embed(x)
        y = self.zh_embed(y)
        h1, h1_n = self.encode(x)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        return F.softmax(d, dim=-1)
