import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CBOWModel(nn.Module):
    def __init__(self, emb_size, emb_dimension, window_size):
        super(CBOWModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.window_size = window_size
        self.u_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        # emb_size = 8934, num_embeddings = 2 * emb_size - 1 = 17867
        self.v_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        # u,v of two embeddings weight
        # question 1: (-0.005, +0.005)  word_embedding  initialization  scope
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        emb_v = []
        for i in range(len(pos_v)):
            emb_v_v = self.u_embeddings(Variable(torch.LongTensor(pos_v[i])))
            emb_v_v_numpy = emb_v_v.data.numpy()
            emb_v_v_numpy = np.sum(emb_v_v_numpy, axis=0)
            emb_v_v_list = emb_v_v_numpy.tolist()
            emb_v.append(emb_v_v_list)
        emb_v = Variable(torch.FloatTensor(emb_v))
        emb_u = self.v_embeddings(Variable(torch.LongTensor(pos_u)))
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))

        neg_emb_v = []
        for i in range(len(neg_v)):
            neg_emb_v_v = self.u_embeddings(Variable(torch.LongTensor(neg_v[i])))
            neg_emb_v_v_numpy = neg_emb_v_v.data.numpy()
            neg_emb_v_v_numpy = np.sum(neg_emb_v_v_numpy, axis=0)
            neg_emb_v_v_list = neg_emb_v_v_numpy.tolist()
            neg_emb_v.append(neg_emb_v_v_list)
        neg_emb_v = Variable(torch.FloatTensor(neg_emb_v))

        neg_emb_u = self.v_embeddings(Variable(torch.LongTensor(neg_u)))
        neg_score = torch.mul(neg_emb_u, neg_emb_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding='utf-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    model = CBOWModel(100, 100, 5)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word, 'xx.txt')


if __name__ == '__main__':
    test()
