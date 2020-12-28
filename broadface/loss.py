import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, scale_factor=64.0, margin=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.criterion = nn.CrossEntropyLoss()

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # input is not l2 normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        loss = self.criterion(logit, label)

        return loss


class BroadFaceArcFace(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale_factor=64.0,
        margin=0.50,
        queue_size=10000,
        compensate=False,
    ):
        super(BroadFaceArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        feature_mb = torch.zeros(0, in_features)
        label_mb = torch.zeros(0, dtype=torch.int64)
        proxy_mb = torch.zeros(0, in_features)
        self.register_buffer("feature_mb", feature_mb)
        self.register_buffer("label_mb", label_mb)
        self.register_buffer("proxy_mb", proxy_mb)

        self.queue_size = queue_size
        self.compensate = compensate

    def update(self, input, label):
        self.feature_mb = torch.cat([self.feature_mb, input.data], dim=0)
        self.label_mb = torch.cat([self.label_mb, label.data], dim=0)
        self.proxy_mb = torch.cat(
            [self.proxy_mb, self.weight.data[label].clone()], dim=0
        )

        over_size = self.feature_mb.shape[0] - self.queue_size
        if over_size > 0:
            self.feature_mb = self.feature_mb[over_size:]
            self.label_mb = self.label_mb[over_size:]
            self.proxy_mb = self.proxy_mb[over_size:]

        assert (
            self.feature_mb.shape[0] == self.label_mb.shape[0] == self.proxy_mb.shape[0]
        )

    def compute_arcface(self, x, y, w):
        cosine = F.linear(F.normalize(x), F.normalize(w))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logit *= self.scale_factor

        ce_loss = self.criterion(logit, y)
        return ce_loss.mean()

    def forward(self, input, label):
        # input is not l2 normalized
        weight_now = self.weight.data[self.label_mb]
        delta_weight = weight_now - self.proxy_mb

        if self.compensate:
            update_feature_mb = (
                self.feature_mb
                + (
                    self.feature_mb.norm(p=2, dim=1, keepdim=True)
                    / self.proxy_mb.norm(p=2, dim=1, keepdim=True)
                )
                * delta_weight
            )
        else:
            update_feature_mb = self.feature_mb

        large_input = torch.cat([update_feature_mb, input.data], dim=0)
        large_label = torch.cat([self.label_mb, label], dim=0)

        batch_loss = self.compute_arcface(input, label, self.weight.data)
        broad_loss = self.compute_arcface(large_input, large_label, self.weight)
        self.update(input, label)

        return batch_loss + broad_loss


if __name__ == "__main__":
    batch_size = 128
    embedding_size = 512
    num_classes = 10000

    arcface_criterion = ArcFace(embedding_size, num_classes)
    broadface_criterion = BroadFaceArcFace(embedding_size, num_classes, queue_size=2000, compensate=True)

    for _ in range(100):
        embedding = torch.randn(batch_size, embedding_size)
        labels = torch.randint(0, num_classes, (batch_size,))

        arcface_loss = arcface_criterion(embedding, labels)
        broadface_loss = broadface_criterion(embedding, labels)
    
        print(arcface_loss.detach())
        print(broadface_loss.detach())