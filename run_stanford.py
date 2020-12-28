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

from utils.dataset import StanfordOnlineProductsMetric
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


def eval_dml(net, loader, K=[1], ep=0):
    net.eval()
    test_iter = tqdm(loader, ncols=80)
    embeddings_all, labels_all = [], []

    test_iter.set_description("[Eval][Epoch %d]" % ep)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)

        embeddings_all = torch.cat(embeddings_all)
        labels_all = torch.cat(labels_all)
        rec = recall(embeddings_all, labels_all, K=K)

        for k, r in zip(K, rec):
            print("[Epoch %d] Recall@%d: [%.4f]\n" % (ep, k, r))

    return rec[0], K, rec


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
    dataset_train = StanfordOnlineProductsMetric(
        opts.data, train=True, transform=train_transform, download=True
    )
    dataset_eval = StanfordOnlineProductsMetric(
        opts.data, train=False, transform=test_transform, download=True
    )

    print("Number of images in Training Set: %d" % len(dataset_train))
    print("Number of images in Test set: %d" % len(dataset_eval))

    loader_train = DataLoader(
        dataset_train,
        batch_size=opts.batch,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    loader_eval = DataLoader(
        dataset_eval,
        batch_size=opts.batch,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )
    
    if opts.loss == loss.ArcFace:
        criterion = loss.ArcFace(
            opts.embedding_size,
            len(dataset_train.classes),
        ).cuda()
    elif opts.loss == loss.BroadFaceArcFace:
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

    val_recall, _, _ = eval_dml(model, loader_eval, opts.recall, 0)
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

        val_recall, val_recall_K, val_recall_all = eval_dml(
            model, loader_eval, opts.recall, epoch
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
