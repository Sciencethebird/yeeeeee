
# coding: utf-8

# ## Import the library

# In[69]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn


# ## layer definition (Need to do!!!)

# In[70]:


def InnerProduct_ForProp(x,W,b):
  
    return y

def InnerProduct_BackProp(dEdy,x,W,b):

    return dEdx,dEdW,dEdb

def L2Loss_BackProp(y,t):

    return dEdx

def Sigmoid_ForProp(x):
 
    return y

def Sigmoid_BackProp(dEdy,x):

    return dEdx

def ReLu_ForProp(x):

    return y

def ReLu_BackProp(dEdy,x):

    return dEdx

def loss_ForProp(y,y_pred):
    #$loss = -np.sum(y_pred*np.log(y))
    return loss

'''
# ## Setup the Parameters and Variables (Can tune that!!!)

# In[71]:


eta =         #learning rate
Data_num = 784      #size of input data   (inputlayer)
W1_num =          #size of first neural (1st hidden layer)
Out_num =         #size of output data  (output layer)
iteration =          #epoch for training   (iteration)
image_num = 60000   #input images
test_num  = 10000   #testing images

## Cross Validation ##
##spilt the training data to 80% train and 20% valid##
train_num = int(image_num*0.8)
valid_num = int(image_num*0.2)


# ## Setup the Data (Create weight array here!!!)

# In[72]:


w_1= (np.random.normal(0,1,Data_num*W1_num)).reshape(Data_num,W1_num)/100
w_out  = (np.random.normal(0,1,W1_num*Out_num)).reshape(W1_num, Out_num)/100
b_1, b_out = randn(1,W1_num)/100,randn(1,Out_num)/100
print("w1 shape:", w_1.shape)
print("w_out shape:", w_out.shape)
print("b_1 shape:", b_1.shape)
print("b_out shape:", b_out.shape)
print(w_out)


# ## Prepare all the data

# ### Load the training data and labels from files

# In[73]:


df = pd.read_csv('fashion-mnist_train_data.csv')
fmnist_train_images = df.as_matrix()
print("Training data:",fmnist_train_images.shape[0])
print("Training data shape:",fmnist_train_images.shape)

df = pd.read_csv('fashion-mnist_test_data.csv')
fmnist_test_images = df.as_matrix()
print("Testing data:",fmnist_test_images.shape[0])
print("Testing data shape:",fmnist_test_images.shape)

df = pd.read_csv('fashion-mnist_train_label.csv')
fmnist_train_label = df.as_matrix()
print("Training label shape:",fmnist_train_label.shape)


# ### Show the 100 testing images

# In[74]:


plt.figure(figsize=(20,20))
for index in range(100):
    image = fmnist_test_images[index].reshape(28,28)
    plt.subplot(10,10,index+1,)
    plt.imshow(image)
plt.show() 


# ### Convert the training labels data type to one hot type

# In[75]:


label_temp = np.zeros((image_num,10), dtype = np.float32)
for i in range(image_num):
    label_temp[i][fmnist_train_label[i][0]] = 1
train_labels_onehot = np.copy(label_temp)
print("Training label shape:",train_labels_onehot.shape)


# ### Separate train_images, train_labels into training and validating 

# In[76]:


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


# ## Execute the Iteration (Need to do!!!)

# In[77]:


valid_accuracy = []
for i in range(iteration):

    # Forward-propagation
  
 
    # Bakcward-propagation

  
    # Parameters Updating (Gradient descent)
  
    # Do cross-validation to evaluate model


    # Get 1-D Prediction array
   # Compare the Prediction and validation


    #print('valid_acc:',v_acc/12000)
    #Calculate the accuracy


# ## Testing Stage

# ### Predict the test images (Do forward propagation again!!!)

# In[81]:


# Forward-propagation



# ### Convert results to csv file (Input the (10000,10) result array!!!)

# In[82]:


# Convert "test_Out_data" (shape: 10000,10) to "test_Prediction" (shape: 10000,1)
test_Prediction      = np.argmax(test_Out_data, axis=1)[:,np.newaxis].reshape(test_num,1)
df = pd.DataFrame(test_Prediction,columns=["Prediction"])
df.to_csv("DL_LAB1_prediction_ID.csv",index=True, index_label="index")


# ## Convert results to csv file

# In[83]:


accuracy = np.array(valid_accuracy)
plt.plot(accuracy, label="$iter-accuracy$")
y_ticks = np.linspace(0, 100, 11)
plt.legend(loc='best')
plt.xlabel('iteration')
plt.axis([0, iteration, 0, 100])
plt.ylabel('accuracy')
plt.show()

'''