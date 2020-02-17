import numpy as np
import pandas as pd
from utils import load_image, BengaliAIDataset, Transform
from model import SENet
import torch
import argparse


parser = argparse.ArgumentParser(description='bengali')
parser.add_argument('--train', default=True, help='train or test')
parser.add_argument('--data-path', default="/kaggle/input/bengaliai-cv19/", help="path to training dataset")
parser.add_argument('--test-fold', default=0, help="which fold for validation")
parser.add_argument('--image-size', default=128, help="input image size for model")

args = parser.parse_args()


def main():
    # config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = SENet().to(device)

    # create dataset
    images = load_image(args.data_path)
    train_pd = pd.read_csv(data_path_original+'train.csv')
    labels = train_pd[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    
    num_train = int(images.shape[0]/5*4)
    train_indices = random.sample(set(range(images.shape[0])), num_train)
    test_indices = [i for i in range(images.shape[0]) if i not in train_indices]

    train_img = np.take(images, train_indices, axis=0)
    test_img = np.take(images, test_indices, axis=0)
    train_labels = np.take(labels, train_indices, axis=0)
    test_labeles = np.take(labels, test_indices, axis=0)


    train_transform = Transform(
    size=(128, 128), threshold=5.,
    sigma=-1., blur_ratio=0.5, noise_ratio=0.5, cutout_ratio=0.5,
    elastic_distortion_ratio=0.5, random_brightness_ratio=0.5,
    piece_affine_ratio=0.5, ssr_ratio=0.5)
    # transform = Transform(size=(image_size, image_size))
    train_dataset = BengaliAIDataset(train_images, train_labels,
                                 transform=train_transform)

    test_transform = Transform(
    size=(128, 128), threshold=5., sigma=-1.)  
    test_dataset = BengaliAIDataset(train_images, test_labels,
                                 transform=train_transform)



if __name__ == "__main__":
    main()