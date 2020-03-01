import numpy as np
import pandas as pd
from utils import load_image, BengaliAIDataset, Transform, load_image_shuffle, rand_bbox
from model import SENet, SEResNeXtBottleneck, SEAttentionNet, SENetHeavyHead
import argparse
import random
import sklearn.metrics
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F



parser = argparse.ArgumentParser(description='bengali')
parser.add_argument('--train', default=True, help='train or test')
parser.add_argument('--data-path', default="/kaggle/input/bengaliai-cv19/", help="path to training dataset")
parser.add_argument('--valid-fold', default=0, type=int, help="which fold for validation")
parser.add_argument('--image-size', default=128, help="input image size for model")
parser.add_argument('--epochs', default=300, type=int, help="epochs to train")
parser.add_argument('--batch-size', default=128, type=int, help="batch size")
parser.add_argument('--save-path', default="/pv/kaggle/bengali/", help="path to save model")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
parser.add_argument('--optim', default='sgd', help="pytorch optimizer")
parser.add_argument('--verbal', default=False, action='store_true')
parser.add_argument('--num-workers', default=16, type=int, help='num of workers for dataloader')
parser.add_argument('--load-model', default=False, action='store_true')
parser.add_argument('--load-model-path', default="/pv/kaggle/bengali/bengali.pt", help='path to model to load')
parser.add_argument('--valid-ratio', default=0.1, type=float, help='how much data is used for validation')
parser.add_argument('--valid-shuffle', default=False, action='store_true',  help='which way to do validation')
parser.add_argument('--net', default=0, type=int, help='which model to use, 0:SEAttentionNet; 1:SENet; 2:SENetHeavyHead')
parser.add_argument('--blur-ratio', default=0, type=float, help='blur ratio')
parser.add_argument('--noise-ratio', default=0, type=float, help='noise ratio')
parser.add_argument('--cutout-ratio', default=0, type=float, help='cutout ratio')
parser.add_argument('--elastic-distortion-ratio', default=0, type=float, help='elastic distortion ratio')
parser.add_argument('--random-brightness-ratio', default=0, type=float, help='random brightness ratio')
parser.add_argument('--piece-affine-ratio', default=0, type=float, help='piece affine ratio')
parser.add_argument('--ssr-ratio', default=0, type=float, help='ssr ratio')
parser.add_argument('--affine', default=False, action='store_true', help='if use image affine')
parser.add_argument('--threshold', default=100, type=float, help='threshold for cutout in image processing')
parser.add_argument('--alpha', default=2.0, type=float, help='alpha value for cutmix')


args = parser.parse_args()


def cutmix(data, labels, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_labels = labels[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    return data, shuffled_labels, lam


def docutmixtrain(model, optimizer, criterion, inputs, labels, alpha):
    model.train()
    optimizer.zero_grad()
    inputs, shuffled_labels, lam = cutmix(inputs, labels, alpha)

    output = model(inputs)
    logit1, logit2, logit3 = output[:,: 168], output[:,168: 168+11], output[:,168+11:]

    loss1 = criterion(logit1, labels[:, 0])
    loss2 = criterion(logit2, labels[:, 1])
    loss3 = criterion(logit3, labels[:, 2])
    loss4 = criterion(logit1, shuffled_labels[:, 0])

    (2*(loss1*lam + loss4*(1-lam))  +loss2+loss3).backward()
    optimizer.step()

    return [loss1.item(), loss2.item(), loss3.item()*lam+loss4.item()*(1-lam)]



def dovalid(model, dataloader, device, criterion):
    model.eval()
    all_labels = [torch.rand(0).type(torch.LongTensor), torch.rand(0).type(torch.LongTensor), torch.rand(0).type(torch.LongTensor)]
    all_preds = [torch.rand(0).type(torch.int64), torch.rand(0).type(torch.int64), torch.rand(0).type(torch.int64)]
    scores = []
    acc = []
    all_loss = [0, 0, 0]
    all_count = 0


    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            output = model(inputs)
        logit1, logit2, logit3 = output[:,: 168], output[:,168: 168+11], output[:,168+11:]
        pred1 = torch.max(logit1, axis=1).indices
        pred2 = torch.max(logit2, axis=1).indices
        pred3 = torch.max(logit3, axis=1).indices

        loss1 = criterion(logit1, labels[:, 0])
        loss2 = criterion(logit2, labels[:, 1])
        loss3 = criterion(logit3, labels[:, 2])

        # save prediction and labels
        preds = [pred1, pred2, pred3]

        for i in range(3):
            all_labels[i] = torch.cat((all_labels[i], labels.cpu()[:, i]), dim=0)
            all_preds[i] = torch.cat((all_preds[i], preds[i].cpu()), dim=0)

        all_count += inputs.shape[0]
        loss = [torch.sum(loss1).item(), torch.sum(loss2).item(), torch.sum(loss3).item()]
        for i in range(3):
            all_loss[i] += loss[i]

    for i in range(3):
        all_labels[i] = all_labels[i].type(torch.int64).numpy()
        all_preds[i] = all_preds[i].numpy()
        scores.append(sklearn.metrics.recall_score(
            all_labels[i], all_preds[i], average='macro'))
        acc.append(sum(all_labels[i] == all_preds[i])/all_labels[0].shape[0])
        all_loss[i] = all_loss[i]/all_count

    return acc, scores, all_loss




def dotrain(model, optimizer, criterion, inputs, labels):
    model.train()
    optimizer.zero_grad()
    output = model(inputs)
    logit1, logit2, logit3 = output[:,: 168], output[:,168: 168+11], output[:,168+11:]
    # logit1 = F.softmax(logit1, dim=1)
    # logit2 = F.softmax(logit2, dim=1)
    # logit3 = F.softmax(logit3, dim=1)
    loss1 = criterion(logit1, labels[:, 0])
    loss2 = criterion(logit2, labels[:, 1])
    loss3 = criterion(logit3, labels[:, 2])
    # pred1 = torch.max(logit1, axis=1).indices
    # pred2 = torch.max(logit2, axis=1).indices
    # pred3 = torch.max(logit3, axis=1).indices
    # acc1, acc2, acc3 = sum(pred1 == labels[:, 0])/pred1.shape[0], sum(pred2 == labels[:, 1])/pred3.shape[0], sum(pred3 == labels[:, 2])/pred3.shape[0] 

    (2*loss1+loss2+loss3).backward()
    optimizer.step()

    return [loss1.item(), loss2.item(), loss3.item()]


def main():

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16).to(device)
    if args.net == 0:
        model = SEAttentionNet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=0.2, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    elif args.net == 1:
        model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=0.2, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    elif args.net == 2:
        model = SENetHeavyHead(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=0.2, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    else:
        print("Undefined Net Option")

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_path))
    model = model.to(device)

    print("Create Model Done")
    # create dataset
    if not args.valid_shuffle:
        train_images = load_image(args.data_path, args.valid_fold)
        valid_images = load_image(args.data_path, args.valid_fold, False)
        df = pd.read_csv(args.data_path+'/train.csv')
        labels = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

        valid_indices = [i+50210*args.valid_fold for i in range(50210)]
        train_indices = [i for i in range(50210*4) if i not in valid_indices]

        train_labels = np.take(labels, train_indices, axis=0)
        valid_labels = np.take(labels, valid_indices, axis=0)
    else:
        images = load_image_shuffle(args.data_path)
        train_pd = pd.read_csv(args.data_path+'/train.csv')
        labels = train_pd[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

        num_train = int(images.shape[0]*(1-args.valid_ratio))
        train_indices = random.sample(set(range(images.shape[0])), num_train)
        valid_indices = [i for i in range(images.shape[0]) if i not in train_indices]

        train_images = np.take(images, train_indices, axis=0)
        valid_images = np.take(images, valid_indices, axis=0)
        train_labels = np.take(labels, train_indices, axis=0)
        valid_labels = np.take(labels, valid_indices, axis=0)
    
    num_train = len(train_indices)

    train_transform = Transform(
        affine = args.affine, size=(128, 128), threshold=args.threshold,
        sigma=-1., blur_ratio=args.blur_ratio, noise_ratio=args.noise_ratio, cutout_ratio=args.cutout_ratio,
        elastic_distortion_ratio=args.elastic_distortion_ratio, random_brightness_ratio=args.random_brightness_ratio,
        piece_affine_ratio=args.piece_affine_ratio, ssr_ratio=args.ssr_ratio)
    # transform = Transform(size=(image_size, image_size))
    train_dataset = BengaliAIDataset(train_images, train_labels,
                                 transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_transform = Transform(
        affine = args.affine, size=(128, 128), threshold=args.threshold, sigma=-1.)  
    valid_dataset = BengaliAIDataset(valid_images, valid_labels,
                                 transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # create train loaders for cutmix
    cutmix_transform = Transform(
        affine = args.affine, size=(128, 128), threshold=args.threshold, sigma=-1.)  

    cutmix_loaders = []
    grapheme_dict = defaultdict(list)
    for i in range(train_labels.shape[0]):
        label_v, label_c = train_labels[i][1], train_labels[i][2]
        grapheme_dict['%2d%2d' % (label_v, label_c)].append(i)
    
    idx, iter_idxes = 0, []
    for _,v in grapheme_dict.items():
        subimages = np.take(train_images, v, axis=0)
        sublabels = np.take(train_labels, v, axis=0)
        cutmix_dataset = BengaliAIDataset(subimages, sublabels,
                                 transform=cutmix_transform)
        cutmix_loaders.append(DataLoader(cutmix_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers))
        iter_idxes += [idx]*(len(cutmix_dataset)//args.batch_size)
        idx += 1
    iter_idxes += [idx]*(len(train_dataset)//args.batch_size)


    print("DataLoader Done")
    # train code

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=6, min_lr=1e-07)

    writer = SummaryWriter()
    best_score = 0

    for epoch in range(args.epochs):
        random.shuffle(iter_idxes)
        all_iters = [iter(loader) for loader in cutmix_loaders]
        all_iters.append(iter(train_loader))

        cutmix_losses = [0, 0, 0]
        train_losses = [0, 0, 0]

        for idx in iter_idxes:
            dataiter = all_iters[idx]
            inputs, labels = next(dataiter)
            inputs, labels = inputs.to(device), labels.to(device)
            if idx < len(all_iters)-1:
                train_loss = dotrain(model, optimizer, criterion, inputs, labels)
                for i in range(3):
                    train_losses[i] += train_loss[i]*inputs.shape[0]
            else:
                cutmix_loss = docutmixtrain(model, optimizer, criterion, inputs, labels, args.alpha)
                for i in range(3):
                    cutmix_losses[i] += cutmix_loss[i]*inputs.shape[0]

        # # train with grapheme root cutmix
        # cutmix_losses = [0, 0, 0]
        # for cutmix_loader in cutmix_loaders:
        #     for inputs, labels in cutmix_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         cutmix_loss = docutmixtrain(model, optimizer, criterion, inputs, labels, args.alpha)
        #         for i in range(3):
        #             cutmix_losses[i] += cutmix_loss[i]*inputs.shape[0]

        # # train with normal data augmentation
        # train_losses = [0, 0, 0]
        # for inputs, labels in train_loader:
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     train_loss = dotrain(model, optimizer, criterion, inputs, labels)
        #     for i in range(3):
        #         train_losses[i] += train_loss[i]*inputs.shape[0]


        train_acc, train_scores, train_loss = dovalid(model, train_loader, device, criterion)
        writer.add_scalars('train acc', {'acc1':train_acc[0], 'acc2': train_acc[1], 'acc3': train_acc[2]}, epoch)
        writer.add_scalars('train score', {'score1':train_scores[0], 'score2': train_scores[1], 'score3': train_scores[2]}, epoch)
        writer.add_scalars('train loss', {'loss1':train_loss[0], 'loss2': train_loss[1], 'loss3': train_loss[2]}, epoch)

        valid_acc, valid_scores, valid_loss = dovalid(model, valid_loader, device, criterion)
        writer.add_scalars('valid acc', {'acc1':valid_acc[0], 'acc2': valid_acc[1], 'acc3': valid_acc[2]}, epoch)
        writer.add_scalars('valid score', {'score1':valid_scores[0], 'score2': valid_scores[1], 'score3': valid_scores[2]}, epoch)
        writer.add_scalars('valid loss', {'loss1':valid_loss[0], 'loss2': valid_loss[1], 'loss3': valid_loss[2]}, epoch)


        print("epoch %d done" % (epoch))

        # save model
        score = (valid_scores[0]*2+valid_scores[1]+valid_scores[2])/4
        if score > best_score:
            torch.save(model.state_dict(), args.save_path+"bengali.pt")
            best_score = score

        train_losses = [loss/num_train for loss in train_losses]
        cutmix_losses = [loss/num_train for loss in cutmix_losses]

        if args.verbal:
            print("Normal Train Losses: %f, %f, %f" % (train_losses[0], train_losses[1], train_losses[2]))
            print("Cutmix Train Losses: %f, %f, %f" % (cutmix_losses[0], cutmix_losses[1], cutmix_losses[2])) 
            print("Train ACC: %f, %f, %f" % (train_acc[0], train_acc[1], train_acc[2]))
            print("Train Scores: %f, %f, %f" % (train_scores[0], train_scores[1], train_scores[2]))
            print("Train Loss: %f, %f, %f" % (train_loss[0], train_loss[1], train_loss[2]))
            print("Valid ACC: %f, %f, %f" % (valid_acc[0], valid_acc[1], valid_acc[2]))
            print("Valid Scores: %f, %f, %f" % (valid_scores[0], valid_scores[1], valid_scores[2]))
            print("Valid Loss: %f, %f, %f" % (valid_loss[0], valid_loss[1], valid_loss[2]))
            print("Best Score: ", best_score)

        scheduler_loss = np.average(train_losses, weights=[2,1,1]) + np.average(cutmix_losses, weights=[2,1,1])
        scheduler.step(scheduler_loss)

if __name__ == "__main__":
    main()