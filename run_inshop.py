import os
import argparse
import random

import torch
import torch.optim as optim

import broadface.loss as loss
import broadface.backbone as backbone
import broadface.embedding as embedding

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import FashionInshop
from utils.common import build_transform, recall


def train(net, loader, optimizer, criterion, ep=0):
    net.train()
    
    train_iter = tqdm(loader, ncols=80)
    loss_all = []
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()
        
        embedding = net(images)
        loss = criterion(embedding, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_iter.set_description(
            "[Train][Epoch %d] Loss: %.5f"
            % (ep, loss.item())
        )

        loss_all.append(loss.item())

    print("[Epoch %d] Loss: %.5f\n" % (ep, torch.Tensor(loss_all).mean()))


def eval_inshop(net, loader_query, loader_gallery, K=[1], ep=0):
    net.eval()
    query_iter = tqdm(loader_query, ncols=80)
    gallery_iter = tqdm(loader_gallery, ncols=80)

    query_embeddings_all, query_labels_all = [], []
    gallery_embeddings_all, gallery_labels_all = [], []

    with torch.no_grad():
        for images, labels in query_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            query_embeddings_all.append(embedding.data)
            query_labels_all.append(labels.data)

        query_embeddings_all = torch.cat(query_embeddings_all)
        query_labels_all = torch.cat(query_labels_all)

        for images, labels in gallery_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            gallery_embeddings_all.append(embedding.data)

        gallery_embeddings_all = torch.cat(gallery_embeddings_all)

    correct_labels = []
    for query_e, query_l in zip(query_embeddings_all, query_labels_all):
        distance = (gallery_embeddings_all - query_e[None]).pow(2).sum(dim=1)
        knn_ind = distance.topk(max(K), dim=0, largest=False, sorted=True)[1]

        query_label_text = loader_query.dataset.data_labels[query_l.item()]
        gallery_label_text = [loader_gallery.dataset.labels[k.item()] for k in knn_ind]

        cl = [query_label_text == g for g in gallery_label_text]
        correct_labels.append(cl)

    correct_labels = torch.FloatTensor(correct_labels)

    recall_k = []
    for k in K:
        correct_k = 100 * (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
        print("[Epoch %d] Recall@%d: [%.4f]\n" % (ep, k, correct_k))

    return recall_k[0], K, recall_k


def build_args():
    parser = argparse.ArgumentParser()
    LookupChoices = type(
        "",
        (argparse.Action,),
        dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])),
    )

    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--load", default=None)
    parser.add_argument(
        "--backbone",
        choices=dict(
            bninception=backbone.BNInception,
            resnet50=backbone.ResNet50,
        ),
        default=backbone.ResNet50,
        action=LookupChoices,
    )
    
    parser.add_argument(
        "--loss",
        choices=dict(
            arcface=loss.ArcFace,
            broadface=loss.BroadFaceArcFace
        ),
        default=loss.ArcFace,
        action=LookupChoices,
    )

    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--queue-size", type=int, default=0)
    parser.add_argument("--compensate", default=False, action="store_true")

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-decay-epochs", type=int, default=[40, 60, 80], nargs="+")
    parser.add_argument("--lr-decay-gamma", default=0.2, type=float)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--recall", default=[1], type=int, nargs="+")

    parser.add_argument("--seed", default=random.randint(1, 1000), type=int)
    parser.add_argument("--data", default="./dataset/")
    parser.add_argument("--save-dir", default="./result")
    opts = parser.parse_args()

    return opts


if __name__ == "__main__":
    opts = build_args()

    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
        set_random_seed(opts.seed)

    base_model = opts.backbone(pretrained=True)
    model = embedding.LinearEmbedding(
            base_model,
            feature_size=base_model.output_size,
            embedding_size=opts.embedding_size,
            l2norm_on_train=False
    ).cuda()

    if opts.load is not None:
        model.load_state_dict(torch.load(opts.load))
        print("Loaded Model from %s" % opts.load)

    train_transform, test_transform = build_transform(base_model)
    dataset_train = FashionInshop(
        opts.data, split="train", transform=train_transform
    )
    dataset_query = FashionInshop(
        opts.data, split="query", transform=test_transform
    )
    dataset_gallery = FashionInshop(
        opts.data, split="gallery", transform=test_transform
    )

    print("Number of images in Training Set: %d" % len(dataset_train))
    print("Number of images in Query set: %d" % len(dataset_query))
    print("Number of images in Gallery set: %d" % len(dataset_gallery))

    loader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=opts.batch,
        pin_memory=True,
        num_workers=8,
    )
    loader_query = DataLoader(
        dataset_query,
        shuffle=False,
        batch_size=opts.batch,
        pin_memory=True,
        num_workers=8,
    )
    loader_gallery = DataLoader(
        dataset_gallery,
        shuffle=False,
        batch_size=opts.batch,
        pin_memory=True,
        num_workers=8,
    )
    
    if opts.loss == loss.ArcFace:
        print("ArcFace")
        criterion = loss.ArcFace(
            opts.embedding_size,
            len(dataset_train.classes),
        ).cuda()
    elif opts.loss == loss.BroadFaceArcFace:
        print("BroadFace")

        criterion = loss.BroadFaceArcFace(
            opts.embedding_size,
            len(dataset_train.classes),
            queue_size=opts.queue_size,
            compensate=opts.compensate
        ).cuda()

    optimizer = optim.Adam(
        [
            {"lr": opts.lr, "params": model.parameters()},
            {"lr": opts.lr, "params": criterion.parameters()},
        ],
        weight_decay=1e-5,
    )

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma
    )

    val_recall, _, _ = eval_inshop(model, loader_query, loader_gallery, opts.recall, ep=0)
    best_rec = val_recall

    if opts.mode == "eval":
        exit(0)

    for epoch in range(1, opts.epochs + 1):
        train(
            model,
            loader_train,
            optimizer,
            criterion,
            ep=epoch,
        )
        lr_scheduler.step()

        val_recall, val_recall_K, val_recall_all = eval_inshop(
            model, loader_query, loader_gallery, opts.recall, epoch
        )

        if best_rec < val_recall:
            best_rec = val_recall
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "best.pth"))

        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, "w") as f:
                f.write("Best Recall@1: %.4f\n" % best_rec)
                f.write("Final Recall@1: %.4f\n" % val_recall)

        print("Best Recall@1: %.4f" % best_rec)
