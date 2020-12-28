import torch
import torchvision.transforms as transforms
import broadface.backbone as backbone

from tqdm import tqdm
from PIL import Image


def recall(embeddings, labels, K=[]):
    knn_inds = []

    evaluation_iter = tqdm(embeddings, ncols=80)
    evaluation_iter.set_description("Measuring recall...")
    for i, e in enumerate(evaluation_iter):
        d = (e.unsqueeze(0) - embeddings).pow(2).sum(dim=1).clamp(min=1e-12)
        d[i] = 0
        knn_ind = d.topk(1 + max(K), dim=0, largest=False, sorted=True)[1][1:]
        knn_inds.append(knn_ind)

    knn_inds = torch.stack(knn_inds, dim=0)

    """
    Check if, knn_inds contain index of query image.
    """

    assert (
        knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)
    ).sum().item() == 0

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    recall_k = []

    for k in K:
        correct_k = 100 * (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
    return recall_k


def fix_batchnorm(net):
    for m in net.modules():
        if (
            isinstance(m, torch.nn.BatchNorm1d)
            or isinstance(m, torch.nn.BatchNorm2d)
            or isinstance(m, torch.nn.BatchNorm3d)
        ):
            m.eval()


def build_transform(model):
    if isinstance(model, backbone.BNInception):
        normalize = transforms.Compose(
            [
                transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255.0),
                transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
            ]
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            transforms.RandomResizedCrop(
                scale=(0.16, 1),
                ratio=(0.75, 1.33),
                size=224,
                interpolation=Image.LANCZOS,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, test_transform
