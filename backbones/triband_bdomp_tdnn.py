import torch
import torch.nn as nn
from quant.modules.ops import Add, Mul, Pow
import torch.nn.utils.prune as prune


class TriBand_BDOMP_TDNN(nn.Module):
    def __init__(self, hidden_size=59, memory_depth=4):
        super(TriBand_BDOMP_TDNN, self).__init__()
        if hidden_size > 59:
            raise ValueError("hidden_size must be <= 59.")

        self.hidden_size = hidden_size
        self.memory_depth = memory_depth
        self.tap_size = memory_depth + 1

        self.add = Add()
        self.mul = Mul()
        self.pow2 = Pow(2)

        self.in_features = 28 * self.tap_size
        self.fc_hidden = nn.Linear(self.in_features, hidden_size, bias=True)
        self.act = nn.Tanh()
        self.fc_out = nn.Linear(hidden_size, 6, bias=True)

    @classmethod
    def apply_unstructured_pruning(cls, model, amount=0.75):
        prune.l1_unstructured(model.fc_hidden, name="weight", amount=amount)

    def _complex_mul(self, a_r, a_i, b_r, b_i):
        real = self.add(self.mul(a_r, b_r), -self.mul(a_i, b_i))
        imag = self.add(self.mul(a_r, b_i), self.mul(a_i, b_r))
        return real, imag

    def _complex_square(self, a_r, a_i):
        a_r2 = self.pow2(a_r)
        a_i2 = self.pow2(a_i)
        real = self.add(a_r2, -a_i2)
        imag = self.add(self.mul(a_r, a_i), self.mul(a_r, a_i))
        return real, imag

    def _mag2(self, a_r, a_i):
        return self.add(self.pow2(a_r), self.pow2(a_i))

    def _feature_extract(self, x):
        i1 = x[..., 0:1]
        q1 = x[..., 1:2]
        i2 = x[..., 2:3]
        q2 = x[..., 3:4]
        i3 = x[..., 4:5]
        q3 = x[..., 5:6]

        mag1 = self._mag2(i1, q1)
        mag2 = self._mag2(i2, q2)
        mag3 = self._mag2(i3, q3)

        f1_r, f1_i = i1, q1
        f2_r, f2_i = self.mul(i1, mag1), self.mul(q1, mag1)
        f3_r, f3_i = i2, q2
        f4_r, f4_i = self.mul(i2, mag2), self.mul(q2, mag2)
        f5_r, f5_i = i3, q3
        f6_r, f6_i = self.mul(i3, mag3), self.mul(q3, mag3)
        f7_r, f7_i = self.mul(i1, mag2), self.mul(q1, mag2)
        f8_r, f8_i = self.mul(i1, mag3), self.mul(q1, mag3)
        f9_r, f9_i = self.mul(i2, mag1), self.mul(q2, mag1)
        f10_r, f10_i = self.mul(i2, mag3), self.mul(q2, mag3)
        f11_r, f11_i = self.mul(i3, mag1), self.mul(q3, mag1)

        x2_sq_r, x2_sq_i = self._complex_square(i2, q2)
        f12_r, f12_i = self._complex_mul(x2_sq_r, x2_sq_i, i1, -q1)
        f13_r, f13_i = self._complex_mul(x2_sq_r, x2_sq_i, i3, -q3)

        x1_sq_r, x1_sq_i = self._complex_square(i1, q1)
        x1_cu_r, x1_cu_i = self._complex_mul(x1_sq_r, x1_sq_i, i1, q1)
        f14_r, f14_i = self._complex_mul(x1_cu_r, x1_cu_i, i3, -q3)

        features = [
            f1_r, f1_i,
            f2_r, f2_i,
            f3_r, f3_i,
            f4_r, f4_i,
            f5_r, f5_i,
            f6_r, f6_i,
            f7_r, f7_i,
            f8_r, f8_i,
            f9_r, f9_i,
            f10_r, f10_i,
            f11_r, f11_i,
            f12_r, f12_i,
            f13_r, f13_i,
            f14_r, f14_i,
        ]

        return torch.cat(features, dim=-1)

    def forward(self, x, h_0=None):
        batch_size = x.size(0)
        frame_length = x.size(1)

        x_feat = self._feature_extract(x)
        pad = torch.zeros((batch_size, self.memory_depth, x_feat.size(-1)), device=x.device, dtype=x.dtype)
        x_pad = torch.cat((pad, x_feat), dim=1)

        windows = x_pad.unfold(dimension=1, size=self.tap_size, step=1)
        windows = windows.contiguous().view(-1, self.in_features)

        out = self.act(self.fc_hidden(windows))
        out = self.fc_out(out)
        out = out.view(batch_size, frame_length, 6)
        return out
