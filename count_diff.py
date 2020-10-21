from PIL import Image
import sys



def compare(dir_a, dir_b, filenames):
    print('comparing %s %s...' % (dir_a, dir_b))


    error_cnt = 0
    for f in filenames:
        #print(f)
        imga = Image.open(dir_a +'/' + f)
        imgb = Image.open(dir_b +'/' + f)
        for h in range(32):
            for w in range(32):
                nowa = imga.getpixel((h, w))
                nowb = imgb.getpixel((h, w))
                for a,b in zip(nowa, nowb):
                    if( abs(a-b) > 8 ):
                        error_cnt += 1

    print(error_cnt, ' images are error!')





def main():
    filenames = []
    with open('./images', 'r') as f:
        filenames = [ line.strip() for line in f.readlines() ]
    compare(sys.argv[1], sys.argv[2], filenames)

if __name__=='__main__':
    main()














