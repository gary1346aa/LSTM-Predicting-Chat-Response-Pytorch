import pandas as pd
import numpy as np
import operator
import random
import scipy as sp
import torch
import time
from torch import optim, nn
import torch.nn.functional as F

from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.svm import SVC
from gensim import models
from numpy import linalg as LA

EMBED_DIM = 256
HIDDEN_DIM = 200
MAX_LEN = 20
BATCH_SIZE = 128

USE_CUDA = True
# USE_CUDA = False

def Variable(x):
    if USE_CUDA:
        return torch.autograd.Variable(x.cuda())
    else:
        return torch.autograd.Variable(x)

cut_programs = np.load('cut_Programs.npy')
cut_questions = np.load('cut_Questions.npy')
W2V = models.Word2Vec.load('W2V_new_256')
# W2V = models.Word2Vec.load('W2V')

episodes = []
for program in cut_programs:
    episodes.extend(program)

episodes = [e for e in episodes if np.max([len(s) for s in e]) <= MAX_LEN]

print(f'# Episodes = {len(episodes)}')

# print([int(np.mean([len(s) for s in x])) for x in episodes])

sentences = []
for epi in episodes:
    sentences.extend(epi)

# sentences = sentences[:100000]

sentences = [s for s in sentences if len(s) > 0]

print(f'# Sentences = {len(sentences)}')

N_train_sen = int(len(sentences) * 0.9)

# train_sentences = sentences[:N_train_sen]

Nvoc = len(W2V.wv.index2word) + 1

print(f'Vocab size = {Nvoc}')

idx_space = W2V.wv.vocab[' '].index
np_space = np.array([idx_space])

def sen2idx(sen):
    for i in range(len(sen)):
        idx = Nvoc - 1
        if sen[i] in W2V:
            idx = W2V.wv.vocab[sen[i]].index
        sen[i] = idx
    return np.array(sen)

def concat_sen(sens):
    res = []
    for s in sens:
        if len(s) == 0:
            continue
        res.append(s)
        res.append(np_space)
    return np.concatenate(res[:-1])

sentences = [sen2idx(s) for s in sentences]

pretrain_emb = np.concatenate([W2V.wv.syn0, np.zeros((1, EMBED_DIM), dtype=np.float32)], axis=0)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(Nvoc, EMBED_DIM, scale_grad_by_freq=True)
        self.embed.weight.data = torch.from_numpy(pretrain_emb)

        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, 1)

        # self.h0 = nn.Parameter(torch.Tensor(1, 1, HIDDEN_DIM))
        # self.c0 = nn.Parameter(torch.Tensor(1, 1, HIDDEN_DIM))
        self.h0 = nn.Parameter(torch.Tensor(1, BATCH_SIZE, HIDDEN_DIM))
        self.c0 = nn.Parameter(torch.Tensor(1, BATCH_SIZE, HIDDEN_DIM))
        nn.init.xavier_uniform(self.h0)
        nn.init.xavier_uniform(self.c0)

        self.M = nn.Parameter(torch.Tensor(HIDDEN_DIM, HIDDEN_DIM))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.xavier_uniform(self.M)
        nn.init.uniform(self.b)

        # self.M = nn.Parameter(torch.randn(EMBED_DIM, HIDDEN_DIM), requires_grad=True)

    def forward(self, seq, lens):
        batch_size = seq.size(0)
        emb = self.embed(seq)
        pck = nn.utils.rnn.pack_padded_sequence(emb, lens, batch_first=True)
        # hiddens = (
            # self.h0.expand(1, batch_size, HIDDEN_DIM),
            # self.c0.expand(1, batch_size, HIDDEN_DIM),
        # )
        hiddens = (self.h0, self.c0)
        # print(pck, hiddens)
        _, (h, _) = self.lstm(pck, hiddens)
        h = h.squeeze(0)
        return h

    def calc_score(self, hc, hr):
        # hc = torch.mm(hc, self.M)
        hc = hc.unsqueeze(1)
        hr = hr.unsqueeze(2)
        res = torch.bmm(hc, hr).view(-1) + self.b
        return res

def pad_sort_seq(seqs, label=None):
    n = len(seqs)
    lens = [len(x) for x in seqs]
    mxlen = max(lens)
    res = np.zeros((n, mxlen), dtype=int)
    for i, s in enumerate(seqs):
        s = np.pad(s, (0, mxlen-len(s)), 'constant')
        res[i] = s

    lens = np.array(lens)
    idx = np.argsort(-lens)
    inv_idx = np.argsort(idx)
    res = res[idx, :]
    new_lens = list(lens[idx])

    var = Variable(torch.from_numpy(res))
    inv_idx = Variable(torch.from_numpy(inv_idx))

    if label is not None:
        label = np.array(label)[idx]
        label = Variable(torch.from_numpy(label))
        return inv_idx, new_lens, var, label
    else:
        return inv_idx, new_lens, var

def gen_training_set(n=100, testing=False):
    if testing == False:
        lb, rb = 0, N_train_sen - 10
    else:
        lb, rb = N_train_sen, len(sentences) - 10

    qno = np.random.randint(lb, rb, size=n)
    slen = np.random.choice([1, 2, 3, 4], size=n, p=[0.1, 0.2, 0.5, 0.2])
    y = np.random.randint(0, 2, size=n)
    anspos = np.random.randint(lb, rb, size=n)
    anspos = y * (qno + slen) + (1-y) * anspos

    q = []
    a = []

    for i in range(n):
        sens = sentences[qno[i]:qno[i]+slen[i]]
        sens = [s for s in sens if len(s) > 0]
        merge = concat_sen(sens)
        q.append(merge)

        ans = sentences[anspos[i]]
        a.append(ans)

    return q, a, y

def gen_eval_set(n=100, testing=False):
    if testing == False:
        lb, rb = 0, N_train_sen - 10
    else:
        lb, rb = N_train_sen, len(sentences) - 10

    qno = np.random.randint(lb, rb, size=n)
    slen = np.random.choice([1, 2, 3, 4], size=n, p=[0.1, 0.2, 0.5, 0.2])
    y = np.random.randint(0, 6, size=n)

    q = []
    a = []

    for i in range(n):
        sens = sentences[qno[i]:qno[i]+slen[i]]
        sens = [s for s in sens if len(s) > 0]
        merge = concat_sen(sens)

        for j in range(6):
            anspos = np.random.randint(lb, rb)
            if j == y[i]:
                anspos = qno[i] + slen[i]
            ans = sentences[anspos]
            q.append(merge)
            a.append(ans)

    return q, a, y

def gen_test_set():
    n = len(cut_questions)
    q = []
    a = []

    for i in range(n):
        merge = concat_sen([sen2idx(s) for s in cut_questions[i][0]])

        for j in range(6):
            ans = sen2idx(cut_questions[i][j+1])
            q.append(merge)
            a.append(ans)

    return q, a

###################

Q_test, A_test = gen_test_set();

encoder = Encoder()
if USE_CUDA:
    encoder = encoder.cuda()
Loss = nn.BCEWithLogitsLoss()
# print([p.norm() for p in encoder.parameters()])

optimizer = optim.Adam(encoder.parameters(), lr=3e-4)
# optimizer = optim.Adagrad(encoder.parameters(), lr=3e-3)

def train_batch(batch_size=BATCH_SIZE, to_train=True, testing=False):
    Q_train, A_train, Y_train = gen_training_set(n=batch_size, testing=testing)
    qidx, qlen, qarr = pad_sort_seq(Q_train)
    aidx, alen, aarr = pad_sort_seq(A_train)
    y = Variable(torch.from_numpy(Y_train.astype(np.float32)))

    hq = encoder(qarr, qlen)[qidx]
    ha = encoder(aarr, alen)[aidx]
    score = encoder.calc_score(hq, ha)
    loss = Loss(score, y)
    loss_val = loss.data.cpu().numpy().mean()
    accu_val = (score * (y-0.5) > 0).data.cpu().numpy().mean()

    if to_train:
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(encoder.parameters(), 20)
        optimizer.step()

    return loss_val, accu_val

def train_epoch(num_batch=200, batch_size=BATCH_SIZE, to_train=True, testing=False):
    loss = 0.0
    accu = 0.0
    for i in range(num_batch):
        bloss, baccu = train_batch(batch_size, to_train, testing)
        loss += bloss
        accu += baccu
    loss /= num_batch
    accu /= num_batch

    return loss, accu

def eval_batch(batch_size=BATCH_SIZE, testing=False):
    Q_train, A_train, Y_train = gen_eval_set(n=batch_size, testing=testing)

    res = []

    n = len(Q_train)
    for i in range(0, n, batch_size):
        lb = i
        rb = min(n, i+batch_size)
        num = rb - lb
        qq = Q_train[lb:rb]
        aa = A_train[lb:rb]
        while len(qq) < batch_size:
            qq.append(qq[-1])
            aa.append(aa[-1])

        qidx, qlen, qarr = pad_sort_seq(qq)
        aidx, alen, aarr = pad_sort_seq(aa)
        hq = encoder(qarr, qlen)[qidx]
        ha = encoder(aarr, alen)[aidx]
        score = encoder.calc_score(hq, ha)
        score = score.data.cpu().numpy()[:num]
        res.extend(list(score))

    res = np.array(res).reshape((batch_size, 6))
    sel = res.argmax(axis=1)
    accu_val = (sel == Y_train).mean()

    return accu_val

def eval_epoch(num_batch=100, batch_size=BATCH_SIZE, testing=False):
    acc = 0.0
    for i in range(num_batch):
        acc += eval_batch(batch_size, testing)
    acc /= num_batch
    return acc

def test_ques(batch_size=BATCH_SIZE):
    res = []

    n = len(Q_test)
    for i in range(0, n, batch_size):
        lb = i
        rb = min(n, i+batch_size)
        num = rb - lb
        qq = Q_test[lb:rb]
        aa = A_test[lb:rb]
        while len(qq) < batch_size:
            qq.append(qq[-1])
            aa.append(aa[-1])

        qidx, qlen, qarr = pad_sort_seq(qq)
        aidx, alen, aarr = pad_sort_seq(aa)
        hq = encoder(qarr, qlen)[qidx]
        ha = encoder(aarr, alen)[aidx]
        score = encoder.calc_score(hq, ha)
        score = score.data.cpu().numpy()[:num]
        res.extend(list(score))

    res = np.array(res).reshape((n // 6, 6))
    sel = res.argmax(axis=1)

    return sel

def save_output(filename):
    sel = test_ques()
    n = len(sel)

    f = open(filename, 'w')
    f.write('Id,Answer\n')
    for i in range(n):
        f.write(f'{i},{sel[i]}\n')
    f.close()

def train(num_epoch=1000):
    print('=== Start Training ===')
    tm = int(time.time())
    for epoch in range(num_epoch):
        loss, accu = train_epoch()
        print(f'Epoch {epoch+1:3d} : Loss = {loss:.5f}, Acc = {accu:.5f}')

        if (epoch+1) % 5 == 0:
            tloss, taccu = train_epoch(num_batch=400, to_train=False, testing=True)
            print(f'Val : Loss = {tloss:.5f}, Acc = {taccu:.5f}')
        if (epoch+1) % 20 == 0:
            taccu = eval_epoch(num_batch=100, testing=False)
            vaccu = eval_epoch(num_batch=100, testing=True)
            print(f'6-class acc: Train = {taccu:.5f}, Val = {vaccu:.5f}')

            fn = f'output/hao_{tm}_{epoch+1}.csv'
            save_output(fn)
            print(f'Saved output at {fn}')

train()

