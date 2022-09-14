import os
import numpy as np
from PIL import Image
from network import forward_propagation

# read data from the npz file
def init_network():
    '''
    helper function
    read trained network from the npz file
    --------------------------------------
    return:
        w1,w2,w3,b1,b2,b3
        parameters of the network
    '''
    network=np.load('network.npz')
    w1=network['w1']
    w2=network['w2']
    w3=network['w3']

    b1=network['b1']
    b2=network['b2']
    b3=network['b3']
    return w1,w2,w3,b1,b2,b3


def predict(img_path):
    '''
    predict the MNIST image with the network
    ----------------------------------------
    Parameter:
        img_path, the path for a single image or a folder contains multiple MNIST image
    -------------------------------------------------------------------
    return:
        the predicted class of the image, if a single input is given

        return a list of predicted classes when a folder is given
    '''
    w1,w2,w3,b1,b2,b3=init_network()
    if os.path.isfile(img_path):
        img=Image.open(img_path)
        A=np.asarray(img).reshape(-1,1)

        return np.argmax(forward_propagation(A,w1,b1,w2,b2,w3,b3)[-1])
    elif os.path.isdir(img_path):
        result=[]
        for i in os.listdir(img_path):
            img=Image.open(img_path+'//'+i)
            A=np.asarray(img).reshape(-1,1)
            ret=np.argmax(forward_propagation(A,w1,b1,w2,b2,w3,b3)[-1])
            result.append(ret)
        return result
    else:
        print('invalid path given')


if __name__=='__main__':
   print(predict(r'C:\Users\Di Yao\OneDrive\Desktop\NN\MNIST - JPG - testing\9\7.jpg'))
   print(predict(r'C:\Users\Di Yao\OneDrive\Desktop\NN\MNIST - JPG - testing\9'))