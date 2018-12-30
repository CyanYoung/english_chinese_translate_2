import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttEncode(nn.Module):
    def __init__(self, en_embed_mat, stack, head):
        super(AttEncode, self).__init__()
        self.stack, self.head = stack, head
        en_vocab_num, en_embed_len = en_embed_mat.size()
        self.embed_len = en_embed_len
        self.en_embed = nn.Embedding(en_vocab_num, en_embed_len, _weight=en_embed_mat)
        self.querys, self.keys, self.vals = [[[nn.Linear(en_embed_len, 200)] * head] * stack] * 3
        self.fuses = [nn.Linear(200 * head, 200)] * stack
        self.lals = [nn.Sequential(nn.Linear(200, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 200))] * stack

    def forward(self, x):
        p = self.make_pos(x)
        x = self.en_embed(x)
        x = x + p
        for i in range(self.stack):
            r = x
            x = self.mul_att(x, x, i)
            x = F.layer_norm(x + r, x.size()[1:])
            r = x
            x = self.lals[i](x)
            x = F.layer_norm(x + r, x.size()[1:])
        return x

    def make_pos(self, x):
        p = torch.zeros(x.size(0), x.size(1), self.embed_len)
        for k in range(x.size(0)):
            off = (x[k] == 0).sum().byte().item()
            for i in range(x.size(1) - off):
                for j in range(self.embed_len):
                    if j % 2:
                        p[k, i + off, j] = torch.sin(i / torch.pow(1e3, i / self.embed_len))
                    else:
                        p[k, i + off, j] = torch.cos(i / torch.pow(1e3, i / self.embed_len))
        return p

    def mul_att(self, h1, h2, i):
        c = list()
        for j in range(self.head):
            q, k, v = self.querys[i][j](h2), self.keys[i][j](h1), self.vals[i][j](h1)
            scale = math.sqrt(k.size(-1))
            d = torch.matmul(q, k.permute(0, 2, 1)) / scale
            a = F.softmax(d, dim=-1)
            c_i = torch.matmul(a, v)
            c.append(c_i)
        x = torch.cat(c, dim=-1)
        return self.fuses[i](x)


class AttDecode(nn.Module):
    def __init__(self, zh_embed_mat, stack, head):
        super(AttDecode, self).__init__()
        self.stack, self.head = stack, head
        zh_vocab_num, zh_embed_len = zh_embed_mat.size()
        self.embed_len = zh_embed_len
        self.zh_embed = nn.Embedding(zh_vocab_num, zh_embed_len, _weight=zh_embed_mat)
        self.querys, self.keys, self.vals = [[[[nn.Linear(zh_embed_len, 200)] * head] * stack] * 2] * 3
        self.fuses = [nn.Linear(200 * head, 200)] * stack
        self.lals = [nn.Sequential(nn.Linear(200, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 200))] * stack

    def forward(self, y, x):
        p = self.make_pos(y)
        y = self.zh_embed(y)
        y = y + p
        for i in range(self.stack):
            r = y
            x = self.mul_att(y, y, i)
            x = F.layer_norm(x + r, x.size()[1:])
            r = x
            x = self.lals[i](x)
            x = F.layer_norm(x + r, x.size()[1:])
        return x

    def make_pos(self, x):
        p = torch.zeros(x.size(0), x.size(1), self.embed_len)
        for k in range(x.size(0)):
            off = (x[k] == 0).sum().byte().item()
            for i in range(x.size(1) - off):
                for j in range(self.embed_len):
                    if j % 2:
                        p[k, i + off, j] = torch.sin(i / torch.pow(1e3, i / self.embed_len))
                    else:
                        p[k, i + off, j] = torch.cos(i / torch.pow(1e3, i / self.embed_len))
        return p

    def mul_att(self, h1, h2, i):
        c = list()
        for j in range(self.head):
            q, k, v = self.querys[i][j](h2), self.keys[i][j](h1), self.vals[i][j](h1)
            scale = math.sqrt(k.size(-1))
            d = torch.matmul(q, k.permute(0, 2, 1)) / scale
            a = F.softmax(d, dim=-1)
            c_i = torch.matmul(a, v)
            c.append(c_i)
        x = torch.cat(c, dim=-1)
        return self.fuses[i](x)
