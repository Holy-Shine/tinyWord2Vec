import pickle as pkl
import torch
import torch.nn.functional as F
from tinyW2V import *
def getKNN(word,K=10):
    model = torch.load('params/params.pkl')
    vector_matrix = model.embeding_matrix
    word_to_ix =  pkl.load(open('params/word_to_ix.pkl','rb'))
    print(word_to_ix.keys())
    ix_to_word =  pkl.load(open('params/ix_to_word.pkl','rb'))
    try:
        ix = word_to_ix[word]
        vector = vector_matrix[ix]
        idx_cosine_sims = [(i,F.cosine_similarity(vector,vector_matrix[i],dim=0).item()) for i in range(vector_matrix.shape[0])]
        idx_cosine_sims = sorted(idx_cosine_sims,key=lambda x:x[1],reverse=True)

        ids = [idx_cosine_sims[i][0] for i in range(K )][:]

        sim_words = [ix_to_word[id] for id in ids]
        print(sim_words)
    except:
        print('%s not in word dict. try another word!'%word)
    print(model.embeding_matrix.shape)


getKNN(u'大学')