import PIL
import numpy as np
import matplotlib.pyplot as plt 

from argparse import ArgumentParser
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import MyDataset, collate_picture

import os
import logging
import datetime

from transform import Resize
#from model import MyModel
from vgg import get_vgg_model
#from vgg19_model import get_vgg19_model
from tqdm import tqdm



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class_name = {
    0 : 'airplane',
    1 : 'automobile',
    2 : 'bird',
    3 : 'cat',
    4 : 'deer',
    5 : 'dog',
    6 : 'frog',
    7 : 'horse',
    8 : 'ship',
    9 : 'truck'
}


def visualize(data, index):
    rgb = np.asarray(data[b'data'][index]).astype("uint8")
    img = rgb.reshape(3,32,32).transpose([1, 2, 0]) 
    plt.imshow(img)
    plt.show()
    #print(img)


def evaluate(answers, predictions):
    count_ans = {'A':0, 'B':0, 'C':0}
    count_pred = {'A':0, 'B':0, 'C':0}
    acc = {'A':0, 'B':0, 'C':0}

    for key in answers.keys():
        count_ans[answers[key]] += 1
        count_pred[predictions[key]] += 1
        if answers[key] == predictions[key]:
            acc[answers[key]] += 1

    print('ans: ', count_ans)
    print('prediction: ', count_pred)
    print('acc', acc)
    print('recallA:', acc['A']/count_ans['A'])
    print('recallB:', acc['B']/count_ans['B'])
    print('recallC:', acc['C']/count_ans['C'])
        
    weight = {'A':0, 'B':0, 'C':0}
    total_num = count_ans['A'] + count_ans['B'] + count_ans['C']
    for k in ['A', 'B', 'C']:
        weight[k] = count_ans[k] / total_num
    # print(weight)

    WAR = 0.0
    for k in ['A', 'B', 'C']:
        ### weight * recall
        WAR += weight[k] * (acc[k]/count_ans[k])
    print(WAR)
    return WAR




def train(args, model, train_loader, start_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for i_epoch in range(args.epoch):
        logging.info('EPOCH: %d' % (start_epoch + i_epoch))

        pbar = tqdm(train_loader)
        running_loss = 0.0    
        logging.info('[Train]')    
        total = 0
        for i, data in enumerate(pbar):
            optimizer.zero_grad()

            batch_sz = len(data['images'])
            #print('batch:', batch_sz)
            ### input/target
            images = torch.stack(data['images'], 0).to(device)
            #imgs = imgs.permute(0, 2, 3, 1)
            labels = torch.stack(data['labels'], 0).to(device)
            
            print('images: ', images.shape)
            print('labels: ', labels.shape)

            y = model(images)
            
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += batch_sz
            pbar.set_description('loss[%f]' % (loss.item()))

            if i % 1000 == 999:
                print('[%d %5d] loss:%03f' % (i_epoch, i, running_loss/100))
                running_loss = 0.0
        ## finish a epoch
        ## save model to specific directory

        logging.info('save model %s %f to ...' % (str(i_epoch+start_epoch)) )
        torch.save({
                'epoch': i_epoch + start_epoch,
                'state_dict': model.state_dict(),
            }, (args.save_path+'model_%s') % (str(i_epoch + start_epoch))
        )




def predict(args, model, predict_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    cnt_total = 0
    cnt_same = 0

    ## output
    output_file = open(args.output, 'w')
    output_file.write('image_id, label\n')
    
    id2tag = {'0':'A', '1':'B', '2':'C'}

    pbar = tqdm(predict_loader)
    for i, data in enumerate(pbar):
        batch_sz = len(data['imgs'])
        imgs = torch.stack(data['imgs'], 0).to(device)
        #imgs = imgs.permute(0, 2, 3, 1)
        #tags = torch.stack(data['tags'], 0).to(device)
        names = data['names']
        with torch.no_grad():
            y = model(imgs)
       
        #print('y:', y)
        for i_ans in range(batch_sz):
            cnt_total += 1
            now_pred = y[i_ans].topk(1)[1].item()
            #print('ans:', tags[i_ans].item(), ' pre:', now_pred)
            output_file.write(names[i_ans]+','+id2tag[str(now_pred)]+'\n')
            #if tags[i_ans].item() == y[i_ans].topk(1)[1].item():
            #    cnt_same += 1
            
    #print('total:', cnt_total, 'same:', cnt_same)

    logging.info('finish predict!')




mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

def get_train_transform(): #mean=mean, std=std, size=0):
    train_transform = transforms.Compose([
        Resize((int(32 * (256 / 224)), int(32 * (256 / 224)))),
        #transforms.Resize(size, size),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(degrees=30),
        #transforms.RandomCrop(size),
        #transforms.CenterCrop(size),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums),
    ])
    return train_transform


def get_predict_transform(size): #mean=mean, std=std, size=0):
    predict_transform = transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.Resize(size, size),
        #transforms.RandomCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums),
    ])
    return predict_transform



def main(args):
    '''
    for i in range(1, 2):
        path = 'data_batch_'+str(i)
        data = unpickle('../data/'+path)
        print(data.keys())
        for index in range(10000):
            print('%35s' % data[b'filenames'][index], end=' ')
            #print(len(data[b'labels'][index]))
            print(class_name[int(data[b'labels'][index])])
            if class_name[int(data[b'labels'][index])] not in count:
                count[class_name[int(data[b'labels'][index])]] = 1
            count[class_name[int(data[b'labels'][index])]] += 1
        print(count)
    '''

    # model
    model = get_vgg_model(out_feature_dim=10)
    print(model)
    
    if args.do_train:
        train_files = []
        for i in range(1, 6):
            train_files.append('../data/data_batch_' + str(i))
        ### trainset
        trainset = MyDataset(
            get_train_transform(), train_files
        )

        train_loader = DataLoader(
            dataset = trainset,
            batch_size = args.batch_size,
            shuffle = True,
            collate_fn = collate_picture
        )
        start_epoch = 0
        train(args, model, train_loader, start_epoch)




def old_main(args):
    
    #model = MyModel(args.size)
    model = get_vgg_model()
    print(model)

    
    ## TODO: Crop method
    if args.do_train:
        start_epoch = 0
        if args.load_model:
            logging.info('loading model...... %s' % (args.load_model))
            checkpoint = torch.load(args.load_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
    

    #### predict
    if args.do_predict:
        transformer = get_predict_transform(512)
        if args.load_model == None and not args.do_train:
            logging.error('load model error')
            exit(1)

        elif args.load_model:
            logging.info('loading model...... %s' % (args.load_model))
            
            checkpoint = torch.load(args.load_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            #model.load_state_dict(torch.load(args.load_model))

        if args.predict_file == None:
            logging.error('No predict file')
            exit(1)


        logging.info('start reading predict files')
        predictset = MyDataset(
            args.predict_file,
            args.data_dir,
            transform = transformer
        )
        print(predictset[0][0].shape)
        print('len', len(predictset))

        predict_loader = DataLoader(
            dataset = predictset,
            batch_size = args.batch_size,
            shuffle = False,
            collate_fn = collate_picture
        )
        predict(args, model, predict_loader)





def parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='data directory')
    parser.add_argument('--data_dir', default='../data/', type=str, help='data directory')
    parser.add_argument('--data_list', default='../data/train.csv', type=str, help='data directory')
    
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epoch', default=10, type=int, help='epoch')
    parser.add_argument('--size', default=1024, type=int, help='image size')
    
    parser.add_argument('--do_predict', action='store_true', help='evaluate')
    parser.add_argument('--load_model', type=str, help='trained model')
    parser.add_argument('--dev_file', default='../data/dev.csv', type=str, help='dev file')
    parser.add_argument('--predict_file', default='../data/dev.csv', type=str, help='the input file for predict')
    parser.add_argument('--output', default='./prediction.csv', type=str, help='my prediction')
    parser.add_argument('--save_path', default='../model', type=str, help='model save path')

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



