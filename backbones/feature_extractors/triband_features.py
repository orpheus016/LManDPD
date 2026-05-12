import torch
import torch.nn as nn
from quant.modules.ops import Add, Mul, Pow


class TriBandFeatureExtractor(nn.Module):
    DEFAULT_FEATURE_FLAGS = {
        "re_x1": True,
        "im_x1": True,
        "re_x1_mag1_sq": True,
        "im_x1_mag1_sq": True,
        "re_x2": True,
        "im_x2": True,
        "re_x2_mag2_sq": True,
        "im_x2_mag2_sq": True,
        "re_x3": True,
        "im_x3": True,
        "re_x3_mag3_sq": True,
        "im_x3_mag3_sq": True,
        "re_x1_mag2_sq": True,
        "im_x1_mag2_sq": True,
        "re_x1_mag3_sq": True,
        "im_x1_mag3_sq": True,
        "re_x2_mag1_sq": True,
        "im_x2_mag1_sq": True,
        "re_x2_mag3_sq": True,
        "im_x2_mag3_sq": True,
        "re_x3_mag1_sq": True,
        "im_x3_mag1_sq": True,
        "re_x2_sq_x1_conj": True,
        "im_x2_sq_x1_conj": True,
        "re_x2_sq_x3_conj": True,
        "im_x2_sq_x3_conj": True,
        "re_x1_cu_x3_conj": True,
        "im_x1_cu_x3_conj": True,
    }

    FEATURE_ORDER = [
        "re_x1",
        "im_x1",
        "re_x1_mag1_sq",
        "im_x1_mag1_sq",
        "re_x2",
        "im_x2",
        "re_x2_mag2_sq",
        "im_x2_mag2_sq",
        "re_x3",
        "im_x3",
        "re_x3_mag3_sq",
        "im_x3_mag3_sq",
        "re_x1_mag2_sq",
        "im_x1_mag2_sq",
        "re_x1_mag3_sq",
        "im_x1_mag3_sq",
        "re_x2_mag1_sq",
        "im_x2_mag1_sq",
        "re_x2_mag3_sq",
        "im_x2_mag3_sq",
        "re_x3_mag1_sq",
        "im_x3_mag1_sq",
        "re_x2_sq_x1_conj",
        "im_x2_sq_x1_conj",
        "re_x2_sq_x3_conj",
        "im_x2_sq_x3_conj",
        "re_x1_cu_x3_conj",
        "im_x1_cu_x3_conj",
    ]

    def __init__(self, feature_flags=None):
        super(TriBandFeatureExtractor, self).__init__()
        self.add = Add()
        self.mul = Mul()
        self.pow2 = Pow(2)

        self.feature_flags = dict(self.DEFAULT_FEATURE_FLAGS)
        if feature_flags is not None:
            unknown = set(feature_flags.keys()) - set(self.DEFAULT_FEATURE_FLAGS.keys())
            if unknown:
                raise ValueError(f"Unknown feature flags: {sorted(unknown)}")
            self.feature_flags.update(feature_flags)

        if not any(self.feature_flags.values()):
            raise ValueError("At least one feature must be enabled.")

        self.num_features = sum(1 for name in self.FEATURE_ORDER if self.feature_flags[name])

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

    def forward(self, x):
        i1 = x[..., 0:1]
        q1 = x[..., 1:2]
        i2 = x[..., 2:3]
        q2 = x[..., 3:4]
        i3 = x[..., 4:5]
        q3 = x[..., 5:6]

        mag1_sq = self._mag2(i1, q1)
        mag2_sq = self._mag2(i2, q2)
        mag3_sq = self._mag2(i3, q3)

        u1_sq_r, u1_sq_i = self._complex_square(i1, q1)
        u2_sq_r, u2_sq_i = self._complex_square(i2, q2)
        u1_cu_r, u1_cu_i = self._complex_mul(u1_sq_r, u1_sq_i, i1, q1)

        u2_sq_x1c_r, u2_sq_x1c_i = self._complex_mul(u2_sq_r, u2_sq_i, i1, -q1)
        u2_sq_x3c_r, u2_sq_x3c_i = self._complex_mul(u2_sq_r, u2_sq_i, i3, -q3)
        u1_cu_x3c_r, u1_cu_x3c_i = self._complex_mul(u1_cu_r, u1_cu_i, i3, -q3)

        features = {
            "re_x1": i1,
            "im_x1": q1,
            "re_x1_mag1_sq": self.mul(i1, mag1_sq),
            "im_x1_mag1_sq": self.mul(q1, mag1_sq),
            "re_x2": i2,
            "im_x2": q2,
            "re_x2_mag2_sq": self.mul(i2, mag2_sq),
            "im_x2_mag2_sq": self.mul(q2, mag2_sq),
            "re_x3": i3,
            "im_x3": q3,
            "re_x3_mag3_sq": self.mul(i3, mag3_sq),
            "im_x3_mag3_sq": self.mul(q3, mag3_sq),
            "re_x1_mag2_sq": self.mul(i1, mag2_sq),
            "im_x1_mag2_sq": self.mul(q1, mag2_sq),
            "re_x1_mag3_sq": self.mul(i1, mag3_sq),
            "im_x1_mag3_sq": self.mul(q1, mag3_sq),
            "re_x2_mag1_sq": self.mul(i2, mag1_sq),
            "im_x2_mag1_sq": self.mul(q2, mag1_sq),
            "re_x2_mag3_sq": self.mul(i2, mag3_sq),
            "im_x2_mag3_sq": self.mul(q2, mag3_sq),
            "re_x3_mag1_sq": self.mul(i3, mag1_sq),
            "im_x3_mag1_sq": self.mul(q3, mag1_sq),
            "re_x2_sq_x1_conj": u2_sq_x1c_r,
            "im_x2_sq_x1_conj": u2_sq_x1c_i,
            "re_x2_sq_x3_conj": u2_sq_x3c_r,
            "im_x2_sq_x3_conj": u2_sq_x3c_i,
            "re_x1_cu_x3_conj": u1_cu_x3c_r,
            "im_x1_cu_x3_conj": u1_cu_x3c_i,
        }

        active = [features[name] for name in self.FEATURE_ORDER if self.feature_flags[name]]
        return torch.cat(active, dim=-1)
