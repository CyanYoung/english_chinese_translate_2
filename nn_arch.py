import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trm(nn.Module):
    def __init__(self, en_embed_mat, zh_embed_mat, pos_mat, head, stack):
        super(Trm, self).__init__()
        self.encode = TrmEncode(en_embed_mat, pos_mat, head, stack)
        self.decode = TrmDecode(zh_embed_mat, pos_mat, head, stack)

    def forward(self, x, y):
        h = self.encode(x)
        return self.decode(y, h)


class TrmEncode(nn.Module):
    def __init__(self, en_embed_mat, pos_mat, head, stack):
        super(TrmEncode, self).__init__()
        en_vocab_num, en_embed_len = en_embed_mat.size()
        self.en_embed = nn.Embedding(en_vocab_num, en_embed_len, _weight=en_embed_mat)
        self.pos = pos_mat
        self.layers = nn.ModuleList([EncodeLayer(en_embed_len, head) for _ in range(stack)])

    def forward(self, x):
        p = self.pos.repeat(x.size(0), 1, 1)
        x = self.en_embed(x)
        x = x + p
        for layer in self.layers:
            x = layer(x)
        return x


class EncodeLayer(nn.Module):
    def __init__(self, embed_len, head):
        super(EncodeLayer, self).__init__()
        self.head = head
        self.qry = nn.Linear(embed_len, 200 * head)
        self.key = nn.Linear(embed_len, 200 * head)
        self.val = nn.Linear(embed_len, 200 * head)
        self.fuse = nn.Linear(200 * head, 200)
        self.lal = nn.Sequential(nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200))

    def mul_att(self, x, y):
        q = self.qry(y).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        k = self.key(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        v = self.val(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        d = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v).transpose(1, 2)
        c = c.contiguous().view(c.size(0), c.size(1), -1)
        return self.fuse(c)

    def forward(self, x):
        r = x
        x = self.mul_att(x, x)
        x = F.layer_norm(x + r, x.size()[1:])
        r = x
        x = self.lal(x)
        return F.layer_norm(x + r, x.size()[1:])


class TrmDecode(nn.Module):
    def __init__(self, zh_embed_mat, pos_mat, head, stack):
        super(TrmDecode, self).__init__()
        zh_vocab_num, zh_embed_len = zh_embed_mat.size()
        self.zh_embed = nn.Embedding(zh_vocab_num, zh_embed_len, _weight=zh_embed_mat)
        self.pos = pos_mat
        self.layers = nn.ModuleList([DecodeLayer(zh_embed_len, head) for _ in range(stack)])
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, zh_vocab_num))

    def forward(self, y, h):
        p = self.pos.repeat(y.size(0), 1, 1)
        y = self.zh_embed(y)
        y = y + p
        for layer in self.layers:
            y = layer(y, h)
        return self.dl(y)


class DecodeLayer(nn.Module):
    def __init__(self, embed_len, head):
        super(DecodeLayer, self).__init__()
        self.head = head
        self.qry = nn.Linear(embed_len, 200 * head)
        self.key = nn.Linear(embed_len, 200 * head)
        self.val = nn.Linear(embed_len, 200 * head)
        self.fuse = nn.Linear(200 * head, 200)
        self.lal = nn.Sequential(nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200))

    def mul_att(self, x, y):
        q = self.qry(y).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        k = self.key(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        v = self.val(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        d = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v).transpose(1, 2)
        c = c.contiguous().view(c.size(0), c.size(1), -1)
        return self.fuse(c)

    def forward(self, y, h):
        r = y
        y = self.mul_att(y, y)
        y = F.layer_norm(y + r, y.size()[1:])
        r = y
        y = self.mul_att(h, y)
        y = F.layer_norm(y + r, y.size()[1:])
        r = y
        y = self.lal(y)
        return F.layer_norm(y + r, y.size()[1:])
