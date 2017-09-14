# Code copied and slightly modified from https://github.com/taolei87/sru

import pkg_resources
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.utils.rnn as R
from cupy.cuda import function
from pynvrtc.compiler import Program


class SRU_Compute(Function):

    SRU_CUDA_INITIALIZED = False
    SRU_FWD_FUNC = None
    SRU_BWD_FUNC = None
    SRU_BiFWD_FUNC = None
    SRU_BiBWD_FUNC = None
    SRU_STREAM = None

    def __init__(self, use_tanh, d_out, bidirectional=False, use_sigmoid=True):
        global SRU_CUDA_INITIALIZED

        super(SRU_Compute, self).__init__()
        self.use_tanh = use_tanh
        self.d_out = d_out
        self.bidirectional = bidirectional
        self.use_sigmoid = use_sigmoid

        if not self.SRU_CUDA_INITIALIZED:
            self.initialize()

    @classmethod
    def initialize(cls):
        code = pkg_resources.resource_string(__name__, 'sru.cu').decode(
            "utf-8")
        prog = Program(code.encode('utf-8'),
                           'sru_prog.cu'.encode('utf-8'))
        ptx = prog.compile()
        mod = function.Module()
        mod.load(bytes(ptx.encode()))
        cls.SRU_FWD_FUNC = mod.get_function('sru_fwd')
        cls.SRU_BWD_FUNC = mod.get_function('sru_bwd')
        cls.SRU_BiFWD_FUNC = mod.get_function('sru_bi_fwd')
        cls.SRU_BiBWD_FUNC = mod.get_function('sru_bi_bwd')

        Stream = namedtuple('Stream', ['ptr'])
        cls.SRU_STREAM = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        cls.SRU_CUDA_INITIALIZED = True

    def forward(self, u, x, bias, lens, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        max_length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (max_length, batch, d * bidir) \
            if x.dim() == 3 else (batch, d * bidir)

        c = x.new(*size)
        h = x.new(*size)

        FUNC = self.SRU_FWD_FUNC if not self.bidirectional else self.SRU_BiFWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            lens.data_ptr(),
            batch,
            max_length,
            d,
            k_,
            h.data_ptr(),
            c.data_ptr(),
            self.use_tanh,
            self.use_sigmoid],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=self.SRU_STREAM
        )

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        self.lens = lens

        if x.dim() == 2:
            last_cell = c
        elif self.bidirectional:
            d_size = (max_length, batch, d)
            index_l = (lens - 1).unsqueeze(0).unsqueeze(-1).expand(*d_size)
            index_r = lens.new(*d_size).zero_()
            index = torch.cat([index_l, index_r], 2)

            last_cell = torch.gather(c, 0, index.long())[0]
        else:
            index = (lens - 1).unsqueeze(0).unsqueeze(-1).expand_as(c)
            last_cell = torch.gather(c, 0, index.long())[0]

        return h, last_cell

    def backward(self, grad_h, grad_last):
        lens = self.lens
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        max_length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d * bidir)
        grad_init = x.new(batch, d * bidir)

        # For DEBUG
        # size = (length, batch, x.size(-1)) if x.dim() == 3 else (batch, x.size(-1))
        # grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        FUNC = self.SRU_BWD_FUNC if not self.bidirectional else self.SRU_BiBWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            c.data_ptr(),
            grad_h.contiguous().data_ptr(),
            grad_last.contiguous().data_ptr(),
            lens.data_ptr(),
            batch,
            max_length,
            d,
            k_,
            grad_u.data_ptr(),
            grad_x.data_ptr() if k_ == 3 else 0,
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.use_tanh,
            self.use_sigmoid],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=self.SRU_STREAM
        )
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


class SRUCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0, use_sigmoid=1,
                 use_tanh=1, bidirectional=False):
        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_tanh = use_tanh
        self.use_sigmoid = use_sigmoid

        out_size = n_out * 2 if bidirectional else n_out
        k = 4 if n_in != out_size else 3
        self.size_per_dir = n_out * k
        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            self.size_per_dir * 2 if bidirectional else self.size_per_dir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out * 4 if bidirectional else n_out * 2
        ))
        self.reset_parameters()

    def reset_parameters(self):
        val_range = (3.0 / self.n_in) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def init_weight(self):
        return self.reset_parameters()

    def set_bias(self, bias_val=0):
        n_out = self.n_out
        if self.bidirectional:
            self.bias.data[n_out * 2:].zero_().add_(bias_val)
        else:
            self.bias.data[n_out:].zero_().add_(bias_val)

    def forward(self, input, c0=None, lens=None):
        assert input.dim() == 2 or input.dim() == 3

        if input.dim() == 1:
            max_len = 1
        else:
            max_len = input.size(0)

        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out * 2
            ).zero_())

        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        u = x_2d.mm(self.weight)

        if lens is None:
            lens = Variable(x.data.new(batch).int().fill_(max_len))

        computer = SRU_Compute(self.use_tanh, n_out, self.bidirectional,
                               use_sigmoid=self.use_sigmoid)

        if self.training and (self.dropout > 0):
            bidir = 2 if self.bidirectional else 1
            mask_h = self.get_dropout_mask_((batch, n_out * bidir),
                                            self.dropout)

            h, c = computer(u, input, self.bias, lens, c0, mask_h)
        else:
            h, c = computer(u, input, self.bias, lens, c0)

        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1 - p).div_(1 - p))


class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0,
                 rnn_dropout=0, batch_first=True, use_sigmoid=1,
                 use_tanh=1, bidirectional=False):
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size * 2 if bidirectional else hidden_size

        for i in range(num_layers):
            l = SRUCell(
                n_in=self.n_in if i == 0 else self.out_size,
                n_out=self.n_out,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                use_tanh=use_tanh,
                use_sigmoid=use_sigmoid,
                bidirectional=bidirectional
            )
            self.rnn_lst.append(l)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()

    def forward(self, input, c0=None, return_hidden=True):
        packed = isinstance(input, R.PackedSequence)
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0

        if packed:
            input, lens = R.pad_packed_sequence(input,
                                                batch_first=self.batch_first)
            lens_list = lens
            lens = Variable(torch.IntTensor(lens))

            if input.data.is_cuda:
                lens = lens.cuda(input.data.get_device())
        else:
            lens = None

        if self.batch_first:
            input = input.transpose(1, 0)

        assert input.dim() == 3  # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = Variable(input.data.new(
                input.size(1), self.n_out * dir_
            ).zero_())
            c0 = [zeros for i in range(self.depth)]
        else:
            assert c0.dim() == 3  # (depth, batch, n_out*dir_)
            c0 = c0.chunk(self.depth, 0)

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i], lens)
            prevx = h
            lstc.append(c)

        if self.batch_first:
            prevx = prevx.transpose(1, 0)

        if packed:
            prevx = R.pack_padded_sequence(prevx, lens_list,
                                           batch_first=self.batch_first)

        if return_hidden:
            return prevx, torch.stack(lstc)
        else:
            return prevx