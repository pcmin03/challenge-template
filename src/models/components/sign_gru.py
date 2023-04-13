import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyPytorchGRU(nn.Module):
    def __init__(self,in_dim, out_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, out_dim, bidirectional=False, batch_first=True)
        self.out_dim = out_dim
        # it might be asier to define your own parameters and remove nn.GRU() ....
        # i am lazy here to use that of nn.GRU()

    def forward(self, x):
        if 1:
            B,L,dim = x.shape
            h = torch.zeros(B, self.out_dim).to(x)
            y2 = []

            for i in range(L):
                x_t = F.linear(x[:,i], self.gru.weight_ih_l0, bias=self.gru.bias_ih_l0)
                h_t = F.linear(h, self.gru.weight_hh_l0, bias=None)
                x_upd, x_reset, x_new = x_t.chunk(3, 1)
                h_upd, h_reset, h_new = h_t.chunk(3, 1)

                reset_gate  = torch.sigmoid(x_reset + h_reset)
                update_gate = torch.sigmoid(x_upd + h_upd)
                new_gate = torch.tanh(x_new + (reset_gate * h_new))
                y = update_gate * h + (1 - update_gate) * new_gate
                y2.append(y)
                h = y
            y2 = torch.stack(y2, 1)
            h2 = h

        return y2
    
