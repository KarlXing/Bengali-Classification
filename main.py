import numpy as np
import pandas as pd
from utils import load_image, BengaliAIDataset, Transform
from model import SENet, SEResNeXtBottleneck
import argparse
import random
import sklearn.metrics

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
parser.add_argument('--batch_size', default=64, type=int, help="batch size")
parser.add_argument('--save-path', default="/pv/kaggle/bengali/", help="path to save model")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
parser.add_argument('--optim', default='sgd', help="pytorch optimizer")
parser.add_argument('--verbal', default=False, action='store_true')
parser.add_argument('--num-workers', default=16, type=int, help='num of workers for dataloader')
parser.add_argument('--load-model', default=False, action='store_true')
parser.add_argument('--load-model-path', default="/pv/kaggle/bengali/bengali.pt", help='path to model to load')

args = parser.parse_args()


def dovalid(model, dataloader, device):
    model.eval()
    all_labels = [torch.rand(0).type(torch.LongTensor), torch.rand(0).type(torch.LongTensor), torch.rand(0).type(torch.LongTensor)]
    all_preds = [torch.rand(0).type(torch.int64), torch.rand(0).type(torch.int64), torch.rand(0).type(torch.int64)]
    scores = []
    acc = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        output = model(inputs)
        logit1, logit2, logit3 = output[:,: 168], output[:,168: 168+11], output[:,168+11:]
        pred1 = torch.max(logit1, axis=1).indices
        pred2 = torch.max(logit2, axis=1).indices
        pred3 = torch.max(logit3, axis=1).indices
        
        # save prediction and labels
        preds = [pred1, pred2, pred3]
        for i in range(3):
            all_labels[i] = torch.cat((all_labels[i], labels[:, i]), dim=0)
            all_preds[i] = torch.cat((all_preds[i], preds[i].cpu()), dim=0)

    for i in range(3):
        all_labels[i] = all_labels[i].type(torch.int64).numpy()
        all_preds[i] = all_preds[i].numpy()
        scores.append(sklearn.metrics.recall_score(
            all_labels[i], all_preds[i], average='macro'))
        acc.append(sum(all_labels[i] == all_preds[i])/all_labels.shape[0])

    return acc, scores




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
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=0.1, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_path))
    model = model.to(device)

    print("Create Model Done")
    # create dataset
    train_images = load_image(args.data_path, args.valid_fold)
    valid_images = load_image(args.data_path, args.valid_fold, False)
    df = pd.read_csv(args.data_path+'/train.csv')
    labels = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    valid_indices = [i+50210*args.valid_fold for i in range(50210)]
    train_indices = [i for i in range(50210*4) if i not in valid_indices]

    train_labels = np.take(labels, train_indices, axis=0)
    valid_labels = np.take(labels, valid_indices, axis=0)


    train_transform = Transform(
    size=(128, 128), threshold=5.,
    sigma=-1., blur_ratio=0, noise_ratio=0.5, cutout_ratio=0.5,
    elastic_distortion_ratio=0.5, random_brightness_ratio=0.5,
    piece_affine_ratio=0.5, ssr_ratio=0.5)
    # transform = Transform(size=(image_size, image_size))
    train_dataset = BengaliAIDataset(train_images, train_labels,
                                 transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_transform = Transform(
    size=(128, 128), threshold=5., sigma=-1.)  
    valid_dataset = BengaliAIDataset(valid_images, valid_labels,
                                 transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("DataLoader Done")
    # train code

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter()
    best_acc = 0

    for i in range(args.epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            train_loss = dotrain(model, optimizer, criterion, inputs, labels)
            writer.add_scalars('train loss', {'loss1':train_loss[0], 'loss2': train_loss[1], 'loss3': train_loss[2]}, i)

        train_acc, train_scores = dovalid(model, train_loader, device)
        writer.add_scalars('train acc', {'acc1':train_acc[0], 'acc2': train_acc[1], 'acc3': train_acc[2]}, i)
        writer.add_scalars('train score', {'score1':train_scores[0], 'score2': train_scores[1], 'score3': train_scores[2]}, i)

        valid_acc, valid_scores = dovalid(model, valid_loader, device)
        writer.add_scalars('valid acc', {'acc1':valid_acc[0], 'acc2': valid_acc[1], 'acc3': valid_acc[2]}, i)
        writer.add_scalars('valid score', {'score1':valid_scores[0], 'score2': valid_scores[1], 'score3': valid_scores[2]}, i)


        print("epoch %d done" % (i))

        if args.verbal:
            print("Train ACC: %f, %f, %f" % (train_acc[0], train_acc[1], train_acc[2]))
            print("Train Scores: %f, %f, %f" % (train_scores[0], train_scores[1], train_scores[2]))
            print("Valid ACC: %f, %f, %f" % (valid_acc[0], valid_acc[1], valid_acc[2]))
            print("Valid Scores: %f, %f, %f" % (valid_scores[0], valid_scores[1], valid_scores[2]))


        # save model
        if valid_acc[0] > best_acc:
            torch.save(model.state_dict(), args.save_path+"bengali.pt")
            best_acc = valid_acc[0]


if __name__ == "__main__":
    main()