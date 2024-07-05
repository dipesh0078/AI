#Implementing gradient descend for linear regression
#y=wx+b
#loss= (y-ypred)**2/N

import numpy as np

#Initialise some parameters
x=np.random.randn(10,1)
y=10*x+np.random.rand()

#Parameters
w=0.0
b=0.0
#Hyperparameters
learning_rate=0.01

#Create gradient descent function
def descend(x,y,w,b,learning_rate):
  dldw=0.0
  dldb=0.0
  N=x.shape[0]

  for xi,yi in zip(x,y):
    
    dldw += -2*(yi-(w*xi+b))*xi
    dldb += -2*(yi-(w*xi+b))*1

    w=w-learning_rate*(1/N)*dldw
    b=b-learning_rate*(1/N)*dldb
    
  return w,b



#itration
for epoch in range(400):
  w,b=descend(x,y,w,b,learning_rate)
  yhat=w*x+b
  loss=np.divide(np.sum((y-yhat)**2,axis=0),x.shape[0])
  print(f'{epoch} loss is {loss}, parameters w:{w},b:{b}')