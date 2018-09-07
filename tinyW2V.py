import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle as pkl

class corpusDataset(Dataset):
    def __init__(self, csvfile, window_size=3, step=5):

        df_org = pd.read_csv(csvfile)
        vocab = set()
        for idx in range(len(df_org)):
            vocab.update(df_org.iloc[idx, 1].lower().split())


        word_to_ix = {word: i for i, word in enumerate(vocab)}
        ix_to_word = {i:word for i, word in enumerate(vocab)}

        # save these 2 parmas
        pkl.dump(word_to_ix,open('params/word_to_ix.pkl','wb'))
        pkl.dump(ix_to_word, open('params/ix_to_word.pkl', 'wb'))

        train_X_org = []
        train_Y_org = []
        for idx in range(len(df_org)):
            words = df_org.iloc[idx, 1].split()
            n_word = len(words)

            for i in range(window_size, n_word - window_size, step):
                context = [word_to_ix[words[i - x]] for x in reversed(range(1, window_size + 1))] + [
                    word_to_ix[words[i + x]] for x in range(1, window_size + 1)]
                target = word_to_ix[words[i]]
                train_X_org.append(context)
                train_Y_org.append(target)
        self.train_X = torch.LongTensor(train_X_org)
        self.train_Y = torch.LongTensor(train_Y_org)
        self.vocab_size = len(vocab)

    def __len__(self):
        return self.train_Y.shape[0]

    def __getitem__(self, idx):
        return (self.train_X[idx], self.train_Y[idx])


class CBOW(nn.Module):

    def __init__(self, w2vdim=50, vocab_size=50, win_size=2):
        super(CBOW,self).__init__()
        self.embeding_matrix = torch.rand(vocab_size, w2vdim)
        self.linear1 = nn.Linear(w2vdim*2*win_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.win_size = win_size
        self.w2vdim = w2vdim
    def forward(self, inputs):
        embdding = F.embedding( inputs,self.embeding_matrix).view(-1, self.win_size*2*self.w2vdim)
        out = F.relu(self.linear1(embdding))
        out = self.linear2(out)
        log_prob = F.log_softmax(out, dim=1)
        return log_prob
