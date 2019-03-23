import time

import pickle as pk

import math

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from nn_arch import Trm

from util import map_item


def get_mask(head, seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len).byte(), diagonal=1)
    mask = torch.unsqueeze(mask, dim=0)
    mask = mask.repeat(head, 1, 1)
    return torch.unsqueeze(mask, dim=0)


def get_pos(seq_len, embed_len):
    pos = torch.zeros(seq_len, embed_len)
    for i in range(seq_len):
        for j in range(embed_len):
            if j % 2:
                pos[i, j] = math.sin(i / math.pow(1e4, j / embed_len))
            else:
                pos[i, j] = math.cos(i / math.pow(1e4, (j - 1) / embed_len))
    return torch.unsqueeze(pos, dim=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detail = False if torch.cuda.is_available() else True

embed_len = 200
seq_len = 50

head, stack = 4, 2

batch_size = 32

path_en_embed = 'feat/en_embed.pkl'
path_zh_embed = 'feat/zh_embed.pkl'
path_zh_word_ind = 'feat/zh_word_ind.pkl'
with open(path_en_embed, 'rb') as f:
    en_embed_mat = pk.load(f)
with open(path_zh_embed, 'rb') as f:
    zh_embed_mat = pk.load(f)
with open(path_zh_word_ind, 'rb') as f:
    zh_word_inds = pk.load(f)

mask_mat = get_mask(head, seq_len).to(device)
pos_mat = get_pos(seq_len, embed_len).to(device)

archs = {'trm': Trm}

paths = {'trm': 'model/trm.pkl'}


def load_feat(path_feats):
    with open(path_feats['en_sent_train'], 'rb') as f:
        train_en_sents = pk.load(f)
    with open(path_feats['zh_sent_train'], 'rb') as f:
        train_zh_sents = pk.load(f)
    with open(path_feats['label_train'], 'rb') as f:
        train_labels = pk.load(f)
    with open(path_feats['en_sent_dev'], 'rb') as f:
        dev_en_sents = pk.load(f)
    with open(path_feats['zh_sent_dev'], 'rb') as f:
        dev_zh_sents = pk.load(f)
    with open(path_feats['label_dev'], 'rb') as f:
        dev_labels = pk.load(f)
    return train_en_sents, train_zh_sents, train_labels, dev_en_sents, dev_zh_sents, dev_labels


def step_print(step, batch_loss, batch_acc):
    print('\n{} {} - loss: {:.3f} - acc: {:.3f}'.format('step', step, batch_loss, batch_acc))


def epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra):
    print('\n{} {} - {:.2f}s - loss: {:.3f} - acc: {:.3f} - val_loss: {:.3f} - val_acc: {:.3f}'.format(
          'epoch', epoch, delta, train_loss, train_acc, dev_loss, dev_acc) + extra)


def tensorize(feats, device):
    tensors = list()
    for feat in feats:
        tensors.append(torch.LongTensor(feat).to(device))
    return tensors


def get_loader(triples):
    en_sents, zh_sents, labels = triples
    triples = TensorDataset(en_sents, zh_sents, labels)
    return DataLoader(triples, batch_size, shuffle=True)


def get_metric(model, loss_func, triples):
    en_sents, zh_sents, labels = triples
    labels = labels.view(-1)
    num = (labels > 0).sum().item()
    prods = model(en_sents, zh_sents)
    prods = prods.view(-1, prods.size(-1))
    preds = torch.max(prods, dim=1)[1]
    loss = loss_func(prods, labels)
    acc = (preds == labels).sum().item()
    return loss, acc, num


def batch_train(models, loss_func, optim, loader, detail):
    total_loss, total_acc, total_num = [0] * 3
    for step, triples in enumerate(loader):
        batch_loss, batch_acc, batch_num = get_metric(models, loss_func, triples)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        total_loss = total_loss + batch_loss.item()
        total_acc, total_num = total_acc + batch_acc, total_num + batch_num
        if detail:
            step_print(step + 1, batch_loss / batch_num, batch_acc / batch_num)
    return total_loss / total_num, total_acc / total_num


def batch_dev(model, loss_func, loader):
    total_loss, total_acc, total_num = [0] * 3
    for step, triples in enumerate(loader):
        batch_loss, batch_acc, batch_num = get_metric(model, loss_func, triples)
        total_loss = total_loss + batch_loss.item()
        total_acc, total_num = total_acc + batch_acc, total_num + batch_num
    return total_loss / total_num, total_acc / total_num


def fit(name, max_epoch, en_embed_mat, zh_embed_mat, path_feats, detail):
    tensors = tensorize(load_feat(path_feats), device)
    bound = int(len(tensors) / 2)
    train_loader, dev_loader = get_loader(tensors[:bound]), get_loader(tensors[bound:])
    en_embed_mat, zh_embed_mat = torch.Tensor(en_embed_mat), torch.Tensor(zh_embed_mat)
    arch = map_item(name, archs)
    model = arch(en_embed_mat, zh_embed_mat, pos_mat, mask_mat, head, stack).to(device)
    loss_func = CrossEntropyLoss(ignore_index=0, reduction='sum')
    learn_rate, min_rate = 1e-3, 1e-5
    min_dev_loss = float('inf')
    trap_count, max_count = 0, 3
    print('\n{}'.format(model))
    train, epoch = True, 0
    while train and epoch < max_epoch:
        epoch = epoch + 1
        model.train()
        optim = Adam(model.parameters(), lr=learn_rate)
        start = time.time()
        train_loss, train_acc = batch_train(model, loss_func, optim, train_loader, detail)
        delta = time.time() - start
        with torch.no_grad():
            model.eval()
            dev_loss, dev_acc = batch_dev(model, loss_func, dev_loader)
        extra = ''
        if dev_loss < min_dev_loss:
            extra = ', val_loss reduce by {:.3f}'.format(min_dev_loss - dev_loss)
            min_dev_loss = dev_loss
            trap_count = 0
            torch.save(model, map_item(name, paths))
        else:
            trap_count = trap_count + 1
            if trap_count > max_count:
                learn_rate = learn_rate / 10
                if learn_rate < min_rate:
                    extra = ', early stop'
                    train = False
                else:
                    extra = ', learn_rate divide by 10'
                    trap_count = 0
        epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra)


if __name__ == '__main__':
    path_feats = dict()
    path_feats['en_sent_train'] = 'feat/en_sent_train.pkl'
    path_feats['zh_sent_train'] = 'feat/zh_sent_train.pkl'
    path_feats['label_train'] = 'feat/label_train.pkl'
    path_feats['en_sent_dev'] = 'feat/en_sent_dev.pkl'
    path_feats['zh_sent_dev'] = 'feat/zh_sent_dev.pkl'
    path_feats['label_dev'] = 'feat/label_dev.pkl'
    fit('trm', 50, en_embed_mat, zh_embed_mat, path_feats, detail)
