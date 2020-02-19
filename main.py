import numpy as np
import pandas as pd
from utils import load_image, BengaliAIDataset, Transform
from model import SENet, SEResNeXtBottleneck
import argparse
import random


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F



parser = argparse.ArgumentParser(description='bengali')
parser.add_argument('--train', default=True, help='train or test')
parser.add_argument('--data-path', default="/kaggle/input/bengaliai-cv19/", help="path to training dataset")
parser.add_argument('--test-fold', default=0, help="which fold for validation")
parser.add_argument('--image-size', default=128, help="input image size for model")
parser.add_argument('--epochs', default=300, help="epochs to train")
parser.add_argument('--batch_size', default=64, help="batch size")
parser.add_argument('--save-path', default="/pv/kaggle/bengali/", help="path to save model")
parser.add_argument('--lr', default=0.001, help="learning rate")
parser.add_argument('--optim', default='sgd', help="pytorch optimizer")

args = parser.parse_args()


def dovalid(model, dataloader, device):
    model.eval()
    num_data = 0
    num_acc1, num_acc2, num_acc3 = 0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        logit1, logit2, logit3 = output[:,: 168], output[:,168: 168+11], output[:,168+11:]
        pred1 = torch.max(logit1, axis=1).indices
        pred2 = torch.max(logit2, axis=1).indices
        pred3 = torch.max(logit3, axis=1).indices
        acc1, acc2, acc3 = sum(pred1 == labels[:, 0]), sum(pred2 == labels[:, 1]), sum(pred3 == labels[:, 2])
        num_acc1 += acc1
        num_acc2 += acc2
        num_acc3 += acc3
        num_data += inputs.shape[0]

    return [num_acc1/num_data, num_acc2/num_data, num_acc3/num_data]




def dotrain(model, optimizer, criterion, inputs, labels):
    model.train()
    optimizer.zero_grad()
    output = model(inputs)
    logit1, logit2, logit3 = output[:,: 168], output[:,168: 168+11], output[:,168+11:]
    logit1 = F.softmax(logit1, dim=1)
    logit2 = F.softmax(logit2, dim=1)
    logit3 = F.softmax(logit3, dim=1)
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
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=186).to(device)

    print("Create Model Done")
    # create dataset
    images = load_image(args.data_path)
    train_pd = pd.read_csv(args.data_path+'/train.csv')
    labels = train_pd[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    
    num_train = int(images.shape[0]/5*4)
    train_indices = random.sample(set(range(images.shape[0])), num_train)
    test_indices = [i for i in range(images.shape[0]) if i not in train_indices]

    train_images = np.take(images, train_indices, axis=0)
    test_images = np.take(images, test_indices, axis=0)
    train_labels = np.take(labels, train_indices, axis=0)
    test_labels = np.take(labels, test_indices, axis=0)


    train_transform = Transform(
    size=(128, 128), threshold=5.,
    sigma=-1., blur_ratio=0.5, noise_ratio=0.5, cutout_ratio=0.5,
    elastic_distortion_ratio=0.5, random_brightness_ratio=0.5,
    piece_affine_ratio=0.5, ssr_ratio=0.5)
    # transform = Transform(size=(image_size, image_size))
    train_dataset = BengaliAIDataset(train_images, train_labels,
                                 transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_transform = Transform(
    size=(128, 128), threshold=5., sigma=-1.)  
    test_dataset = BengaliAIDataset(test_images, test_labels,
                                 transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

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

        train_acc = dovalid(model, train_loader, device)
        writer.add_scalars('train acc', {'acc1':train_acc[0], 'acc2': train_acc[1], 'acc3': train_acc[2]}, i)

        test_acc = dovalid(model, test_loader, device)
        writer.add_scalars('valid acc', {'acc1':test_acc[0], 'acc2': test_acc[1], 'acc3': test_acc[2]}, i)

        print("epoch %d done" % (i))

        # save model
        if test_acc[0] > best_acc:
            torch.save(model.state_dict(), args.save_path+"bengali.pt")
            best_acc = test_acc[0]


if __name__ == "__main__":
    main()