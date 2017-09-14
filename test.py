import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import rnn

from torchsru import SRU


def test():
    batch_first = False
    sru = SRU(4, 4,
              batch_first=batch_first,
              bidirectional=True,
              use_sigmoid=False).cuda()
    sru.reset_parameters()
    x = Variable(torch.randn(3, 2, 4)).cuda()
    lens = [3, 3]

    h1, c1 = sru(x)

    pack = rnn.pack_padded_sequence(x, lens, batch_first=batch_first)
    h2, c2 = sru(pack)
    h2, _ = rnn.pad_packed_sequence(h2, batch_first=batch_first)

    x = torch.cat([x, Variable(x.data.new(1, 2, 4).zero_())])
    pack = rnn.pack_padded_sequence(x, lens, batch_first=batch_first)
    h3, c3 = sru(pack)
    h3, _ = rnn.pad_packed_sequence(h3, batch_first=batch_first)

    h3.mean().backward()

    h_eq = (h1 == h2) == (h1 == h3)
    c_eq = (c1 == c2) == (c1 == c3)

    assert h_eq.sum().data[0] == np.prod(h_eq.size()) and \
           c_eq.sum().data[0] == np.prod(c_eq.size())