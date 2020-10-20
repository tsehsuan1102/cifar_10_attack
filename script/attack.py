from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
import PIL
import numpy as np
import matplotlib.pyplot as plt 

from argparse import ArgumentParser
from torch import nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset, my_collate_fn
import torchattacks

import os
import logging
import datetime
from tqdm import tqdm
from const import idx2label, label2idx
from utils import imshow, image_folder_custom_label


model_list = ['resnet20_cifar10', 'resnet1001_cifar10', 'resnet1202_cifar10', 'nin_cifar10', 'preresnet20_cifar10']
model_name = model_list[3]



def get_transform(size):
    mean_nums = [0.5, 0.5, 0.5]
    std_nums  = [0.25, 0.25, 0.25]
    
    predict_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean_nums, std_nums),
    ])
    return predict_transform

def save_image(tensor, filename):
    unloader = transforms.ToPILImage()

    dir_name = './results/' + model_name + '/'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    image.save(dir_name + filename)











def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    ## load original data
    transform = get_transform(32)
    origin_data  = image_folder_custom_label(root_dir='./origin', transform=transform, idx2label=classes)
    origin_loader = torch.utils.data.DataLoader(origin_data, batch_size=1, shuffle=False)

    
    ## load model
    model = ptcv_get_model(model_name, pretrained=True)
    print(model)
    
    
    pbar = tqdm(origin_loader)
    for i, (images, labels) in enumerate(pbar):

        print(i)
    


    ## attack
    atk = torchattacks.PGD(model, eps = 8/255, alpha = 2/255, steps=4)
    adversarial_images = atk(images, labels)
    
    print(adversarial_images.shape)    





def parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--filenames', default='./filenames', type=str, help='input filenames')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_argument()

    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=loglevel,
        datefmt='%m-%d %H:%M:%S'
    )
    main(args)













