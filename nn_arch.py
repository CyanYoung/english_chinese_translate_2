import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrmEncode(nn.Module):
    def __init__(self, embed_len, head):
        super(TrmEncode, self).__init__()
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
    def __init__(self, embed_len, head):
        super(TrmDecode, self).__init__()
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

    def forward(self, x, y):
        r = y
        y = self.mul_att(y, y)
        y = F.layer_norm(y + r, y.size()[1:])
        r = y
        y = self.mul_att(y, x)
        y = F.layer_norm(y + r, y.size()[1:])
        r = y
        y = self.lal(y)
        return F.layer_norm(y + r, y.size()[1:])
