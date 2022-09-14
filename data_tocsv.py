import numpy as np
from PIL import Image
import os
from tqdm.auto import tqdm
def data_tocsv(dir,name):
    '''
    read images from JPG to Numpy Arrays, store the arrays to a csv file
    where the first column is the label, and the rest columns are X values in (1,785) shape
    --------------------------------------------------------------------------
    Parameters:
        dir: string
            path of the dataset folder
        name: string
            name of the csv file eg: train_images.csv
    '''
    first=np.zeros((1,785))
    dir=dir+'/{}'
    for target in range(10):
        path=dir.format(str(target))
        for i in tqdm(os.listdir(path)):
            img=Image.open(path+'\\'+i)
            array=np.asarray(img)
            array=array.reshape(1,784)
            array=np.append(target,array)
            first=np.vstack((first,array))
    np.savetxt(f"{name}", first[1:], delimiter=",")

if __name__=="__main__":
    data_tocsv('./MNIST - JPG - training','./files/train_images.csv')
    data_tocsv('./MNIST - JPG - testing','./files/test_images.csv')