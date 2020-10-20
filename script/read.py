import PIL
import numpy as np
import matplotlib.pyplot as plt 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict




class_name = {
    0 : 'airplain',
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

count = {}












def visualize(data, index):
    rgb = np.asarray(data[b'data'][index]).astype("uint8")
    img = rgb.reshape(3,32,32).transpose([1, 2, 0]) 
    #plt.imshow(img)
    #plt.show()
    #print(img)



def main():
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



if __name__=='__main__':
    main()











