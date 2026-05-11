import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from backbones.feature_extractors.triband_features import TriBandFeatureExtractor


class TriBand_BDOMP_TDNN(nn.Module):
    def __init__(self, hidden_size=59, memory_depth=4, feature_flags=None):
        super(TriBand_BDOMP_TDNN, self).__init__()
        if hidden_size > 59:
            raise ValueError("hidden_size must be <= 59.")
        if memory_depth != 4:
            raise ValueError("memory_depth must be 4 for this feature set.")

        self.hidden_size = hidden_size
        self.memory_depth = memory_depth
        self.tap_size = memory_depth + 1

        self.feature_extractor = TriBandFeatureExtractor(feature_flags=feature_flags)
        self.in_features = self.feature_extractor.num_features * self.tap_size
        self.fc_hidden = nn.Linear(self.in_features, hidden_size, bias=True)
        self.act = nn.Tanh()
        self.fc_out = nn.Linear(hidden_size, 6, bias=True)

    @classmethod
    def apply_unstructured_pruning(cls, model, amount=0.75):
        prune.l1_unstructured(model.fc_hidden, name="weight", amount=amount)

    def forward(self, x, h_0=None):
        batch_size = x.size(0)
        frame_length = x.size(1)

        x_feat = self.feature_extractor(x)
        pad = torch.zeros((batch_size, self.memory_depth, x_feat.size(-1)), device=x.device, dtype=x.dtype)
        x_pad = torch.cat((pad, x_feat), dim=1)

        windows = x_pad.unfold(dimension=1, size=self.tap_size, step=1)
        windows = windows.contiguous().view(-1, self.in_features)

        out = self.act(self.fc_hidden(windows))
        out = self.fc_out(out)
        out = out.view(batch_size, frame_length, 6)
        return out
