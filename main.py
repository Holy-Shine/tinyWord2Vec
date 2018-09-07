from tinyW2V import *
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn

from optparse import OptionParser


def get_user_params():

    try:
        opt = OptionParser()
        opt.add_option('--batch_size',
                       dest='batch_size',
                       type=int,
                       help='set mini-batch\'s size. default: 40',
                       default=40)
        opt.add_option('--epoch',
                       dest='epochs',
                       type=int,
                       help='set training epoch. default: 1',
                       default=1)
        opt.add_option('--embed',
                       dest='embed_dim',
                       type=int,
                       help='set word vector dim. default: 50',
                       default=50)
        opt.add_option('--winsize',
                       dest='window_size',
                       type=int,
                       help='set CBOW\'s number of nerghbor words. default: 2',
                       default=2)
        opt.add_option('--step',
                       dest='step',
                       type=int,
                       help='set moving window\'s step. default: 4',
                       default=4)

        (options, args) = opt.parse_args()

        error_msg = []
        step = options.step
        window_size = options.window_size
        embed_dim = options.embed_dim
        batch_size = options.batch_size
        epochs = options.epochs

        user_parmas={'step':step,
                     'window_size':window_size,
                     'embed_dim':embed_dim,
                     'batch_size':batch_size,
                     'epochs':epochs}
        return user_parmas
    except:
        pass




def main():
    user_parms = get_user_params()
    step = user_parms['step']
    window_size = user_parms['window_size']
    embed_dim = user_parms['embed_dim']
    batch_size = user_parms['batch_size']
    epochs = user_parms['epochs']
    print('***********************training setting***************************')
    print('*epoch: %d'%epochs)
    print('*batch_size: %d' % batch_size)
    print('*embed_dim: %d' % embed_dim)
    print('*window_size: %d' % window_size)
    print('*step: %d' % step)
    print('******************************************************************')
    dataset = corpusDataset('sample.csv',step=step,window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = CBOW(w2vdim=embed_dim, vocab_size=dataset.vocab_size, win_size=window_size)

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for i, i_batched in enumerate(dataloader):

            optimizer.zero_grad()
            X, Y = i_batched

            log_probs = model(X)

            loss = criterion(log_probs, Y)

            if i % 19 == 0:
                print('loss after 20 batches:{}'.format(loss))
            loss.backward()
            optimizer.step()

    torch.save(model, 'params/params.pkl')

if __name__=='__main__':
    main()


