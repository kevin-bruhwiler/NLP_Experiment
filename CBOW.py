import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pickle
import re
import random

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_len):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_len*embedding_dim, 256)
        self.fc2 = nn.Linear(256, vocab_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def saveEmbeddings(self):
        np.save("data/embeddings.npy", self.embedding.weight.data.cpu().numpy())

    def forward(self, input):
        emb = self.embedding(input).view(input.size()[0],-1)
        out = self.relu(self.fc1(emb))
        out = self.softmax(self.fc2(out))
        return out

def toVariable(input):
    var = Variable(torch.LongTensor(input))
    if torch.cuda.is_available():
        var = var.cuda()
    return var

def ixToVariable(input, word_to_ix):
    ixs = [[word_to_ix[word] for word in words] for words in input]
    var = Variable(torch.LongTensor(ixs))
    if torch.cuda.is_available():
        var = var.cuda()
    return var

def ixToVector(ix, size):
    vec = [0]*size
    print(vec)
    vec[ix] = 1
    return vec

def loadText():
    remove = re.compile('[,\.!?:,;-]')
    with open("data/text.txt", 'r') as txt:
        data = txt.read().replace('\n', ' ').lower()
    data = remove.sub('', data)
    with open("data/words.pkl", "wb") as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    return data

def main():
    batch_size = 32
    embedding_dim = 2
    text = loadText().split()
    vocab = set(text)
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    model = CBOW(vocab_size, embedding_dim, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if torch.cuda.is_available():
        model.cuda()

    data = []
    for i in range(2, len(text)-2):
        input = [text[i-1], text[i-2], text[i+1], text[i+2]]
        target = text[i]
        data.append((input, target))

    try:
        for it in range(250):
            total_loss = 0
            count = 0
            loss_func = nn.NLLLoss()
            random.shuffle(data)
            for i in range(0, len(data), batch_size+1):
                x = ixToVariable([d[0] for d in data[i:i+batch_size]], word_to_ix)
                y = toVariable([word_to_ix[d[1]] for d in data[i:i+batch_size]])
                model.zero_grad()

                out = model.forward(x)
                loss = loss_func(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.cpu().data[0]
                count += 1
        print("Iteration: ", it, "      Loss: ", total_loss/count)
    except KeyboardInterrupt:
        pass
    model.saveEmbeddings()
    with open("data/word_to_ix.pkl", "wb") as file:
        pickle.dump(word_to_ix, file, pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    print("Beginning....")
    main()
