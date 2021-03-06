import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trm(nn.Module):
    def __init__(self, en_embed_mat, zh_embed_mat, pos_mat, att_mat, head, stack):
        super(Trm, self).__init__()
        self.encode = TrmEncode(en_embed_mat, pos_mat, head, stack)
        self.decode = TrmDecode(zh_embed_mat, pos_mat, att_mat, head, stack)

    def forward(self, x, y):
        h = self.encode(x)
        return self.decode(y, h, x)


class TrmEncode(nn.Module):
    def __init__(self, en_embed_mat, pos_mat, head, stack):
        super(TrmEncode, self).__init__()
        en_vocab_num, en_embed_len = en_embed_mat.size()
        self.en_embed = nn.Embedding(en_vocab_num, en_embed_len, _weight=en_embed_mat)
        self.pos, self.head = pos_mat, head
        self.layers = nn.ModuleList([EncodeLayer(en_embed_len, head) for _ in range(stack)])

    def get_pad(self, x):
        seq_len = x.size(1)
        pad = (x == 0)
        for _ in range(2):
            pad = torch.unsqueeze(pad, dim=1)
        return pad.repeat(1, self.head, seq_len, 1)

    def forward(self, x):
        p = self.pos.repeat(x.size(0), 1, 1)
        m = self.get_pad(x)
        x = self.en_embed(x)
        x = x + p
        for layer in self.layers:
            x = layer(x, m)
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
        self.lns = nn.ModuleList([nn.LayerNorm(200) for _ in range(2)])

    def mul_att(self, x, y, m):
        q = self.qry(y).view(y.size(0), y.size(1), self.head, -1).transpose(1, 2)
        k = self.key(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        v = self.val(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        d = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        d = d.masked_fill(m, -float('inf'))
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v).transpose(1, 2)
        c = c.contiguous().view(c.size(0), c.size(1), -1)
        return self.fuse(c)

    def forward(self, x, m):
        r = x
        x = self.mul_att(x, x, m)
        x = self.lns[0](x + r)
        r = x
        x = self.lal(x)
        return self.lns[1](x + r)


class TrmDecode(nn.Module):
    def __init__(self, zh_embed_mat, pos_mat, att_mat, head, stack):
        super(TrmDecode, self).__init__()
        zh_vocab_num, zh_embed_len = zh_embed_mat.size()
        self.zh_embed = nn.Embedding(zh_vocab_num, zh_embed_len, _weight=zh_embed_mat)
        self.pos, self.att, self.head = pos_mat, att_mat, head
        self.layers = nn.ModuleList([DecodeLayer(zh_embed_len, head) for _ in range(stack)])
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, zh_vocab_num))

    def get_pad(self, x):
        seq_len = x.size(1)
        pad = (x == 0)
        for _ in range(2):
            pad = torch.unsqueeze(pad, dim=1)
        return pad.repeat(1, self.head, seq_len, 1)

    def forward(self, y, h, x):
        p = self.pos.repeat(x.size(0), 1, 1)
        mx = self.get_pad(x)
        my = self.att.repeat(y.size(0), 1, 1, 1) | self.get_pad(y)
        y = self.zh_embed(y)
        y = y + p
        for layer in self.layers:
            y = layer(y, h, my, mx)
        return self.dl(y)


class DecodeLayer(nn.Module):
    def __init__(self, embed_len, head):
        super(DecodeLayer, self).__init__()
        self.head = head
        self.qrys = nn.ModuleList([nn.Linear(embed_len, 200 * head) for _ in range(2)])
        self.keys = nn.ModuleList([nn.Linear(embed_len, 200 * head) for _ in range(2)])
        self.vals = nn.ModuleList([nn.Linear(embed_len, 200 * head) for _ in range(2)])
        self.fuses = nn.ModuleList([nn.Linear(200 * head, 200) for _ in range(2)])
        self.lal = nn.Sequential(nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200))
        self.lns = nn.ModuleList([nn.LayerNorm(200) for _ in range(3)])

    def mul_att(self, x, y, m, i):
        q = self.qrys[i](y).view(y.size(0), y.size(1), self.head, -1).transpose(1, 2)
        k = self.keys[i](x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        v = self.vals[i](x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        d = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        d = d.masked_fill(m, -float('inf'))
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v).transpose(1, 2)
        c = c.contiguous().view(c.size(0), c.size(1), -1)
        return self.fuses[i](c)

    def forward(self, y, x, my, mx):
        r = y
        y = self.mul_att(y, y, my, 0)
        y = self.lns[0](y + r)
        r = y
        y = self.mul_att(x, y, mx, 1)
        y = self.lns[1](y + r)
        r = y
        y = self.lal(y)
        return self.lns[2](y + r)
