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


model_list = [
        'resnet20_cifar10', 'resnet1001_cifar10', 'resnet1202_cifar10',
        'nin_cifar10',
        'preresnet20_cifar10', 'preresnet1001_cifar10',
        'pyramidnet164_a270_bn_cifar10',
        'densenet250_k24_bc_cifar10',
        'xdensenet40_2_k36_bc_cifar10',
        'ror3_164_cifar10',
    ]
#model_name = model_list[0]
global model_name


def get_transform(size):
    #mean_nums = [0.5, 0.5, 0.5]
    #std_nums  = [0.25, 0.25, 0.25]
    
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




def count(model, data_loader):
    pbar = tqdm(data_loader)
    
    count = 0
    accu = 0
    for data in pbar:
        count += 1
        images = data['imgs']
        labels = data['labels']
        filenames = data['filenames']
        pbar.set_description('process %s' % (filenames[0]))
        y = model(images)
        pred = torch.argmax(y, 1)
        if pred.item() == labels[0].item():
            accu += 1

    return count, accu




def evaluate(model, origin_loader):
    origin_count = count(model, origin_loader)

    print('origin data accuracy: %d' % origin_count[1])
    print('-'*150)



def main(args):
    global model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    ## load original data
    transform = get_transform(32)
    #origin_data  = image_folder_custom_label(root_dir='./origin', transform=transform, idx2label=classes)
    #origin_loader = torch.utils.data.DataLoader(origin_data, batch_size=1, shuffle=False)
    origin_data = MyDataset(
        filedir = args.filedir,
        filenames = args.filenames,
        transform=transform
    )
    origin_loader = DataLoader(
        origin_data,
        batch_size=1,
        shuffle=False,
        collate_fn=my_collate_fn
    )
  
    
    for i in range(len(model_list)):
        model_name = model_list[i]
        ## load model
        model = ptcv_get_model(model_name, pretrained=True)
        
        logging.info('Model: %s\n' % (model_name))

        evaluate(model, origin_loader)




def parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--filenames', default='./filenames', type=str, help='input filenames')
    parser.add_argument('--filedir', default='./origin/', type=str, help='input filenames')
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














