import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image
from pathlib import Path
from const import idx2label, label2idx

class MyDataset(Dataset):
    def __init__(self, filenames, transform=None):
        with open(filenames) as f:
            self.filenames = [ line.strip() for line in f.readlines() ]
        self.transform = transform
        
    def __getitem__(self, idx):
        now = self.filenames[idx]
        #os.path.splitext(os.path.split(path)[1])
        filename = os.path.split(now)[1]
        label_name = Path(now).stem[:-1]
        if label_name[-1] == '1':
            label_name = label_name[:-1]
        

        #print(label_name, label2idx[label_name])
        image = Image.open(now).convert('RGB')
        image = self.transform(image)
        
        return image, torch.tensor(label2idx[label_name], dtype=torch.long), filename

    def __len__(self):
        return len(self.filenames)


def my_collate_fn(samples):
    #batchsize*channel*h*w
    imgs = []
    labels = []
    filenames = []
    for sample in samples:
        imgs.append(sample[0])
        labels.append(sample[1])
        filenames.append(sample[2])
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)

    batch = {'imgs': imgs, 'labels': labels, 'filenames': filenames}
    return batch









