""" A PyTorch implementation of CBOW word embedding mechanism. """

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable

CONTEXT_SIZE = 2
EMBED_DIM = 32
HIDDEN_SIZE = 128
LR = 0.0001
NUM_EPOCHS = 10

corpus = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
print(corpus)
# Assign indices to words
vocab = set(corpus)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Assemble bags
data = list()
for i in range(2, len(corpus) - 2):
    # Context, target
    bow = ([corpus[i - 2], corpus[i - 1], corpus[i + 1], corpus[i + 2]], corpus[i])
    data.append(bow)


class CBOW(nn.Module):
    def __init__(self, vocab_size, context_size, embed_dim, hidden_size):
        super(CBOW, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.linear_1 = nn.Linear(2 * context_size * embed_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_data):
        embeds = self.embed_layer(input_data).view((1, -1))
        output = F.relu(self.linear_1(embeds))
        output = F.log_softmax(self.linear_2(output))
        return output


# Helper function
def context_to_tensor(context, idx_dict):
    """ Converts context list to tensor. """
    context_idx = [idx_dict[word] for word in context]
    return Variable(torch.LongTensor(context_idx))

# Define training utilities
model = CBOW(len(vocab), CONTEXT_SIZE, EMBED_DIM, HIDDEN_SIZE)
loss_function = nn.NLLLoss()
optimizer = opt.Adam(model.parameters(), lr=LR)

# Training loop
for e in range(NUM_EPOCHS):
    total_loss = torch.FloatTensor([0])
    for bag in data:
        # Get data and labels
        context_data = context_to_tensor(bag[0], word_to_idx)
        target_data = Variable(torch.LongTensor([word_to_idx[bag[1]]]))
        # Do forward pass
        model.zero_grad()
        prediction = model(context_data)
        loss = loss_function(prediction, target_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    # Bookkeeping
    print('Epoch: %d | Loss: %f.4' % (e, total_loss.numpy()))
