import numpy as np
from tqdm.auto import tqdm


# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
np.random.seed(1)



def init_setup():
    #three layers perception
    w1=(np.random.randn(256,784))
    b1=(np.random.randn(256,1))
    #second layer
    w2=(np.random.randn(64,256))
    b2=(np.random.randn(64,1))
    #third layer
    w3=(np.random.randn(10,64))
    b3=(np.random.randn(10,1))
    return w1,b1,w2,b2,w3,b3


def activate(A):
    Z=np.maximum(0,A)
    return Z

def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

def forward_propagation(A,w1,b1,w2,b2,w3,b3):

    z1=w1@A+b1
    A1=activate(z1)
    z2=w2@A1+b2
    A2=activate(z2)

    z3=w3@A2+b3
    #print("z3",z3.shape)
    # prob=softmax(z3,axis= 0)
    prob=softmax(z3)
    #print("prob",prob.shape)
    prob=prob.reshape(-1,1)
    
    return z1,A1,z2,A2,z3,prob



def one_hot(Y):
    one_hot   =np.zeros((10, 1),dtype = np.float_)
    
    one_hot[Y] = 1

    return one_hot


def delta_cross_entropy(prob,Y):
    assert prob.shape == Y.shape
    grad = prob - Y
    return grad



def back_propagation(A,z1,A1:np.ndarray,z2,A2:np.ndarray,z3,prob,w1,w2:np.ndarray,w3,Y:np.ndarray,lr:float):

    dz3=  delta_cross_entropy(prob,Y)

    dw3=  dz3@A2.T

    db3=  np.copy(dz3)
    dz2=  np.multiply(ReLU_deriv(z2),w3.T @ dz3)

    dw2 =  dz2 @ A1.T
    db2 =  np.copy(dz2)
    dz1 =  np.multiply(ReLU_deriv(z1),w2.T@ dz2) 

    dw1 =  dz1@A.T
    db1 =  np.copy(dz1)
    
    return db1,dw1,dw2,db2,dw3,db3
     

def ReLU_deriv(Z):
    Z[Z > 0] = 1
    Z[Z <=0] = 0
    return Z 

def step(lr,w1,b1,w2,b2,w3,b3,dw1,db1,dw2,db2,dw3,db3):
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1    
    w2 = w2 - lr * dw2  
    b2 = b2 - lr * db2
    w3 = w3 - lr * dw3 
    b3 = b3 - lr * db3
    
    return w1,b1,w2,b2,w3,b3

def evaluate(X_test,y_test,w1,b1,w2,b2,w3,b3):
    hit = 0
    for i,A in enumerate(X_test):
        prob    = forward_propagation(A.reshape(-1,1),w1,b1,w2,b2,w3,b3)[-1]
        if prob.argmax() == y_test[i]:
            hit += 1
    return hit/len(X)
    
    

def learn(X,y,X_test,Y_test,epcohs):
    lr=0.001
    w1,b1,w2,b2,w3,b3=init_setup()
    #read data from csv
    for epoch in range(epcohs):
        # lr=lr/10
        bar = tqdm(range(len(X[1:])))
        for i in bar:
            #first column is the label
            Y = one_hot(y[i])
            #1:785 is values
            A = X[i]
            #reshape the values so it can be passed in to the network
            A = A.reshape(784,1)
            
            z1,A1,z2,A2,z3,prob    = forward_propagation(A,w1,b1,w2,b2,w3,b3)

            db1,dw1,dw2,db2,dw3,db3= back_propagation(A,z1,A1,z2,A2,z3,prob,w1,w2,w3,Y,lr)
            #print(dw1)
            w1,b1,w2,b2,w3,b3      = step(lr,w1,b1,w2,b2,w3,b3,dw1,db1,dw2,db2,dw3,db3)
        acc = evaluate(X_test,Y_test,w1,b1,w2,b2,w3,b3)
        print("epoch",epoch,"accuracy",acc)

    return  w1,b1,w2,b2,w3,b3



if __name__ == "__main__":
    # mnist = datasets.MNIST(root = "./dataset/",download = True,transform=lambda x : np.asarray(x)/255.0)
    mnist=np.genfromtxt('files/train_images.csv', delimiter=',')
    np.random.shuffle(mnist)
    labels=mnist[:,0]
    values=mnist[:,1:]/255
    #read test imges csv
    mnist_test=np.genfromtxt('files/test_images.csv', delimiter=',')
    np.random.shuffle(mnist_test)
    labels_test=mnist[:,0]
    values_test=mnist[:,1:]/255

    X  = np.array(values).reshape(-1,784)
    Y  = np.array(labels,dtype = "int")
    X_test  = np.array(values_test).reshape(-1,784)
    Y_test  = np.array(labels_test,dtype = "int")
    
    w1,b1,w2,b2,w3,b3 = learn(X,Y,X_test,Y_test,10)
    np.savez('files/network',w1=w1,w2=w2,w3=w3,b1=b1,b2=b2,b3=b3)

