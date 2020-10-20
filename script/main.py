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
#import torchattacks

import os
import logging
import datetime
from tqdm import tqdm

from const import idx2label, label2idx


model_list = ['resnet20_cifar10', 'resnet1001_cifar10', 'resnet1202_cifar10', 'nin_cifar10', 'preresnet20_cifar10']
model_name = model_list[3]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_transform(size):
    mean_nums = [0.5, 0.5, 0.5]
    std_nums  = [0.25, 0.25, 0.25]
    
    predict_transform = transforms.Compose([
        transforms.Resize(size, size),
        transforms.ToTensor(),
        #transforms.Normalize(mean_nums, std_nums),
    ])
    return predict_transform

def fgsm_attack(image, epsilon, data_grad):
    #print('atk:')
    #print(image.shape)
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #print(perturbed_image.shape)
    # Return the perturbed image
    return perturbed_image























def diff(imga, imgb):






def save_image(tensor, filename):
    unloader = transforms.ToPILImage()

    dir_name = './results/' + model_name + '/'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    image.save(dir_name + filename)






def attack(model, data_loader):
    #x = torch.randn(1, 3, 224, 224)
    # batch_size, channel, h, w

    pbar = tqdm(data_loader)

    origin = 0
    perb = 0
    diff = 0

    for i, batch in enumerate(pbar):
        #print(i, batch)
        imgs = batch['imgs']
        labels = batch['labels']
        print('\ntruth: ', labels)
        imgs.requires_grad = True
        
        y = model(imgs)
        predict = torch.argmax(y, 1)
        
        print('origin: ', predict)
        if labels[0] != predict[0]:
            for pic, name in zip(perturbed_batch, batch['filenames']):
                save_image(pic, name)
            continue
        

        loss_f = torch.nn.CrossEntropyLoss()
        loss = loss_f(y, labels)
        loss.backward()
        # Collect datagrad
        data_grad = imgs.grad.data
        eps = 0.03

        perturbed_batch = torch.stack([fgsm_attack(x, eps, data_grad[idx]) for idx,x in enumerate(imgs)])       
        perturbed_y = model(perturbed_batch)
        ### save
        for pic, name in zip(perturbed_batch, batch['filenames']):
            #save_image(pic, idx2label[torch.argmax(perturbed_y).item()] + '_' + name)
            save_image(pic, name)



        perturbed_predict = torch.argmax(perturbed_y, 1)
        print('perb', perturbed_predict)
        

        for ans, ori_pre, per_pre in zip(labels, predict, perturbed_predict):
            if ori_pre == ans:
                origin += 1
            if per_pre == ans:
                perb += 1
            if ori_pre != per_pre:
                diff += 1

    print('origin ', origin)
    print('perb   ', perb)
    print('diff   ', diff)

    with open('./results/' + model_name + '/accu.txt', 'w') as f:
        f.write(str(origin)+'\n'+str(perb)+'\n')


#def eval()



def main(args):
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ptcv_get_model(model_name, pretrained=True)
    print(model)
    
    trans = get_transform(32)
    target_dataset = MyDataset(
        filenames = args.filenames,
        transform = trans
    )
    data_loader = DataLoader(
        dataset = target_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        collate_fn = my_collate_fn
    )

    attack(model, data_loader)
    #atk = torchattacks.PGD(model, eps = 8/255, alpha = 2/255, steps=4)
    #adversarial_images = atk(images, labels)
    





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













