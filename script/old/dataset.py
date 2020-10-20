import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def visualize(data, index):
    rgb = np.asarray(data[b'data'][index]).astype("uint8")
    img = rgb.reshape(3,32,32).transpose([1, 2, 0]) 
    plt.imshow(img)
    plt.show()
    #print(img)


class MyDataset(Dataset):
    def __init__(self, transform, filenames):
        self.files = []
        self.image = []
        self.label = []
        for filename in filenames:
            data = unpickle(filename)
            self.files.extend(data[b'filenames'])
            self.image.extend(data[b'data'])
            self.label.extend(data[b'labels'])
        
        self.transform = transform
        #print(len(self.files))
        #print(len(self.labels))
        #print(type(self.labels[0]))
        #print(len(self.labels[0]))       

        rgb = np.asarray(self.image).astype("uint8")
        img = [torch.from_numpy(x.reshape(3, 32, 32).transpose([1, 2, 0])) for x in rgb]

        self.image = torch.stack(img)
        

    def __getitem__(self, idx):
        #image = self.transform(image)

        label = self.label[idx]
        image = self.image[idx]
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.files)


def collate_picture(samples):
    #batchsize*channel*h*w
    batch_image = []
    batch_label = []
    for sample in samples:
        batch_image.append(sample[0])
        batch_label.append(sample[1])
    batch = {'images': batch_image, 'labels': batch_label}
    return batch











