"""
Description: Tri-band Quantized GRU (TriBand_QGRU) backbone
"""

import torch
from torch import nn


class TriBand_QGRU(nn.Module):
    def __init__(self, hidden_size, output_size=6, num_layers=1, bidirectional=False, batch_first=True,
                 bias=True):
        super(TriBand_QGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 12
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = True
        self.bias = bias

        # Instantiate NN Layers
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=True,
                          bias=self.bias)

        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=self.output_size,
                                bias=self.bias)

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])
            if 'weight_ih_l0' in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(param[i * self.hidden_size:(i + 1) * self.hidden_size, :])

        for name, param in self.fc_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, h_0):
        if x.size(-1) != 6:
            raise ValueError("TriBand_QGRU expects input with 6 features (I1, Q1, I2, Q2, I3, Q3).")
        # Feature Extraction
        i1 = torch.unsqueeze(x[..., 0], dim=-1)
        q1 = torch.unsqueeze(x[..., 1], dim=-1)
        i2 = torch.unsqueeze(x[..., 2], dim=-1)
        q2 = torch.unsqueeze(x[..., 3], dim=-1)
        i3 = torch.unsqueeze(x[..., 4], dim=-1)
        q3 = torch.unsqueeze(x[..., 5], dim=-1)

        amp2_1 = torch.pow(i1, 2) + torch.pow(q1, 2)
        amp2_2 = torch.pow(i2, 2) + torch.pow(q2, 2)
        amp2_3 = torch.pow(i3, 2) + torch.pow(q3, 2)

        amp4_1 = torch.pow(amp2_1, 2)
        amp4_2 = torch.pow(amp2_2, 2)
        amp4_3 = torch.pow(amp2_3, 2)

        x = torch.cat((i1, q1, amp2_1, amp4_1,
                       i2, q2, amp2_2, amp4_2,
                       i3, q3, amp2_3, amp4_3), dim=-1)

        # Regressor
        out, _ = self.rnn(x, h_0)
        out = self.fc_out(out)
        return out
