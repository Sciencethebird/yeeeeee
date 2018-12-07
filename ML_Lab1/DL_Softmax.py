
# coding: utf-8

# ## Import the library

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn


# ## layer definition (Need to do!!!)

# In[3]:


def InnerProduct_For(x,W,b):
    y = x*W+b
    return y

def InnerProduct_Back(dEdy,x,W,b):
    dEdx = dEdy * W.transpose() 
    dEdW = x.transpose()*dEdy
    dEdb = dEdy 
    
    return dEdx, dEdW, dEdb

def Softmax_For(x):
    exps = np.exp(x)
    softmax = exps / np.sum(exps)
    return softmax

def Softmax_Back(y,t):
    dEdx = t-y
    return dEdx

def Sigmoid_For(x):
    return 1/( 1+ np.exp(-x) )

def Sigmoid_Back(dEdy ,x):

    dEdx = np.array(  np.exp(-x) / np.square( 1+np.exp(-x) )  )  *  np.array(dEdy)
    return np.matrix(dEdx)

def loss_For(y,y_pred):

    loss =  - np.sum(  np.array(y) * np.log( np.array(y_pred) )  ) 
    return loss
'''
def ReLu_For(x):

    return y

def ReLu_Back(dEdy,x):
            
    return dEdx
'''

# ## Setup the Parameters and Variables (Can tune that!!!)

# In[4]:


eta =  0.000001      #learning rate       ###########lolololololol
Data_num = 784      #size of input data   (inputlayer, 784 pixels)
W1_num = 15         #size of first neural (1st hidden layer)
Out_num = 10        #size of output data  (output layer)
iteration = 500       #epoch for training   (iteration)
image_num = 60000   #input images
test_num  = 10000   #testing images

## Cross Validation ##
##spilt the training data to 80% train and 20% valid##
train_num = int(image_num*0.8)
valid_num = int(image_num*0.2)


# ## Setup the Data (Create weight array here!!!)

# In[5]:


w_1= (np.random.normal(0,0.1,Data_num*W1_num)).reshape(Data_num,W1_num)
w_out  = (np.random.normal(0,0.1,W1_num*Out_num)).reshape(W1_num, Out_num)
b_1, b_out = randn(1,W1_num),randn(1,Out_num)
print("w1 shape:", w_1.shape)
print("w_out shape:", w_out.shape)
print("b_1 shape:", b_1.shape)
print("b_out shape:", b_out.shape)
print(w_1)


# ## Prepare all the data

# ### Load the training data and labels from files

# In[6]:


df = pd.read_csv('fashion-mnist_train_data.csv')
fmnist_train_images = df.values
print("Training data:",fmnist_train_images.shape[0])
print("Training data shape:",fmnist_train_images.shape)

df = pd.read_csv('fashion-mnist_test_data.csv')
fmnist_test_images = df.values
print("Testing data:",fmnist_test_images.shape[0])
print("Testing data shape:",fmnist_test_images.shape)

df = pd.read_csv('fashion-mnist_train_label.csv')
fmnist_train_label = df.values
print("Training labels shape:",fmnist_train_label.shape)



# ### Show the 100 testing images

# In[7]:

'''
plt.figure(figsize=(20,20))
for index in range(100):
    image = fmnist_test_images[index].reshape(28,28)
    plt.subplot(10,10,index+1)
    plt.imshow(image)
plt.show() 

'''
# ### Convert the training labels data type to one hot type

# In[8]:


label_temp = np.zeros((image_num,10), dtype = np.float32)
for i in range(image_num):
    label_temp[i][fmnist_train_label[i][0]] = 1
train_labels_onehot = np.copy(label_temp)
print("\nTraining labels shape:",train_labels_onehot.shape)


# ### Separate train_images, train_labels into training and validating 

# In[13]:

train_data_img = np.copy(fmnist_train_images[:train_num,:])
train_data_lab = np.copy(train_labels_onehot[:train_num,:])
valid_data_img = np.copy(fmnist_train_images[train_num:,:])
valid_data_lab = np.copy(train_labels_onehot[train_num:,:])

# Normalize the input data between (0,1)
train_data_img = train_data_img/255.
valid_data_img = valid_data_img/255.
test_data_img = fmnist_test_images/255.

print("Train images shape:",train_data_img.shape)
print("Train labels shape:",train_data_lab.shape)
print("Valid images shape:",valid_data_img.shape)
print("Valid labels shape:",valid_data_lab.shape)
print("Test  images shape:",test_data_img.shape)
print(train_data_lab)


# ## Execute the Iteration (Need to do!!!)

# In[12]:


valid_accuracy = []

for i in range(iteration):

    # Forward-propagation
        img  = np.matrix(train_data_img)
        w1   = np.matrix(w_1)
        b1   = np.matrix(b_1)
        wout = np.matrix(w_out)
        bout = np.matrix(b_out)
        
        z1 = InnerProduct_For(img, w1, b1) # save for sigmoid prime (backpropagation)
        zout = Sigmoid_For(z1)

        #Softmax
        yp = Softmax_For(InnerProduct_For(zout, wout, bout)) 

        #Cross Entropy Loss
        loss = loss_For(train_data_lab, yp) 
        print('Loss', loss)
        valid_accuracy.append(loss)

    # Bakcward-propagation
  
        soft_back = Softmax_Back(train_data_lab, yp) # y-t

        #output layer
        dEdx, dEdW, dEdb = InnerProduct_Back(soft_back, zout, w_out, b_out) #InnerProduct_Back(dEdy,x,W,b)
        bout_grad = np.average(dEdb, axis = 0)
        wout_grad = np.average(dEdW, axis = 0)
        #input layer
        dEdx, dEdW, dEdb = InnerProduct_Back(Sigmoid_Back(dEdx , z1), img, w_1, b_1)
        b1_grad = np.average(dEdb, axis = 0)
        w1_grad = np.average(dEdW, axis = 0)
  

    # Parameters Updating (Gradient descent)
        w_out = w_out - eta * wout_grad
        b_out = b_out - eta * bout_grad
        w_1 = w_1 - eta * w1_grad
        b_1 = b_1 - eta * b1_grad

        
print(valid_accuracy)     
plt.plot(np.arange(0, iteration, 1), valid_accuracy, label = 'accuracy')
plt.show()

    
    
    
    # Do cross-validation to evaluate model
    
    # Get 1-D Prediction array
    
   
    # Compare the Prediction and validation
    
    #Calculate the accuracy
    #valid_accuracy.append(accuracy)

'''
# ## Testing Stage

# ### Predict the test images (Do forward propagation again!!!)

# In[10]:


# Forward-propagation


# ### Convert results to csv file (Input the (10000,10) result array!!!)

# In[12]:


# Convert "test_Out_data" (shape: 10000,10) to "test_Prediction" (shape: 10000,1)

test_Prediction      = np.argmax(test_Out_data, axis=1)[:,np.newaxis].reshape(test_num,1)
df = pd.DataFrame(test_Prediction,columns=["Prediction"])
df.to_csv("DL_LAB1_prediction_ID.csv",index=True, index_label="index")



# ## Convert results to csv file

# In[16]:


accuracy = np.array(valid_accuracy)
plt.plot(accuracy, label="$iter-accuracy$")
y_ticks = np.linspace(0, 100, 11)
plt.legend(loc='best')
plt.xlabel('iteration')
plt.axis([0, iteration, 0, 100])
plt.ylabel('accuracy')
plt.show()
'''
