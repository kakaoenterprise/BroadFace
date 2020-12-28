import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEmbedding(nn.Module):
    def __init__(
        self, base, feature_size=512, embedding_size=128, l2norm_on_train=True
    ):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(feature_size, embedding_size)
        self.l2norm_on_train = l2norm_on_train

    def forward(self, x):
        feat = self.base(x)
        feat = feat.view(x.size(0), -1)
        embedding = self.linear(feat)

        if self.training and (not self.l2norm_on_train):
            return embedding

        embedding = F.normalize(embedding, dim=1, p=2)
        return embedding
