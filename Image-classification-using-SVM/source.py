#!/usr/bin/env python
# coding: utf-8

# ### Data set preparation

# In[1]:


import numpy as np
import os
from pathlib import Path
from keras.preprocessing import image


# In[2]:


p=Path("images")
dirs=p.glob("*")
labels_dict={"dalmatia":0, "dollar_bil":1, "pizz":2, "soccer_bal":3, "sunflowe":4}

image_data=[]
labels=[]

for folder_dir in dirs:
   # print(folder_name)
    label=str(folder_dir).split("\\")[-1][:-1]
    # print(label)
    
    for img_path in folder_dir.glob("*.jpg"):
        img=image.load_img(img_path, target_size=(100,100))
        img_array=image.img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[label])


# ### Convert this into a numpy array

# In[3]:


image_data=np.array(image_data, dtype='float32')/255.0
labels=np.array(labels)
# print(image_data.shape,labels.shape)


# ### Randomly Shuffle our data!

# In[4]:


import random
combined=list(zip(image_data,labels))
random.shuffle(combined)

#unzip
image_data[:],labels[:]=zip(*combined)


# ### Visualize this data

# In[5]:


def drawImg(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    return


# In[6]:


for i in range(10):
    drawImg(image_data[i])


# ### SVM Classifier

# In[7]:


class SVM:
    """SVM Class, Author : Prateek Narang"""
    def __init__(self,C=1.0):
        self.C = C
        self.W = 0
        self.b = 0
        
    def hingeLoss(self,W,b,X,Y):
        loss  = 0.0
        
        loss += .5*np.dot(W,W.T)
        
        m = X.shape[0]
        
        for i in range(m):
            ti = Y[i]*(np.dot(W,X[i].T)+b)
            loss += self.C *max(0,(1-ti))
            
        return loss[0][0]
    
    def fit(self,X,Y,batch_size=50,learning_rate=0.001,maxItr=500):
        
        no_of_features = X.shape[1]
        no_of_samples = X.shape[0]
        
        n = learning_rate
        c = self.C
        
        #Init the model parameters
        W = np.zeros((1,no_of_features))
        bias = 0
        
        #Initial Loss
        
        #Training from here...
        # Weight and Bias update rule that we discussed!
        losses = []
        
        for i in range(maxItr):
            #Training Loop
            
            l = self.hingeLoss(W,bias,X,Y)
            losses.append(l)
            ids = np.arange(no_of_samples)
            np.random.shuffle(ids)
            
            #Batch Gradient Descent(Paper) with random shuffling
            for batch_start in range(0,no_of_samples,batch_size):
                #Assume 0 gradient for the batch
                gradw = 0
                gradb = 0
                
                #Iterate over all examples in the mini batch
                for j in range(batch_start,batch_start+batch_size):
                    if j<no_of_samples:
                        i = ids[j]
                        ti =  Y[i]*(np.dot(W,X[i].T)+bias)
                        
                        if ti>1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += c*Y[i]*X[i]
                            gradb += c*Y[i]
                            
                #Gradient for the batch is ready! Update W,B
                W = W - n*W + n*gradw
                bias = bias + n*gradb
                
        
        self.W = W
        self.b = bias
        return W,bias,losses


# ### Converting data for one-for-one classification

# In[8]:


M = image_data.shape[0] 
image_data = image_data.reshape(M,-1)
# print(image_data.shape)
# print(labels.shape)


# In[9]:


CLASSES = len(np.unique(labels))
# print(CLASSES)                               #no of classes 


# In[10]:


def classWiseData(x,y):
    data = {}
    
    for i in range(CLASSES):
        data[i] = []
        
    for i in range(x.shape[0]):
        data[y[i]].append(x[i])
    
    for k in data.keys():
        data[k] = np.array(data[k])
        
    return data


# In[11]:


data = classWiseData(image_data,labels)


# In[12]:


def getDataPairForSVM(d1,d2):
    """Combines Data of two classes into a signle matrix"""
    
    l1,l2 = d1.shape[0],d2.shape[0]
    
    samples = l1+l2
    features = d1.shape[1]
    
    data_pair = np.zeros((samples,features))
    data_labels = np.zeros((samples,))
    
    data_pair[:l1,:] = d1
    data_pair[l1:,:] = d2
    
    data_labels[:l1] = -1
    data_labels[l1:] = +1
    
    return data_pair,data_labels


# ### Training NC2 SVM Part

# In[13]:


import matplotlib.pyplot as plt
mySVM  = SVM()
xp, yp  = getDataPairForSVM(data[0],data[1])
w,b,loss  = mySVM.fit(xp,yp,learning_rate=0.00001,maxItr=1000)
#print(loss)
plt.plot(loss)


# In[14]:


def trainSVMs(x,y):
    
    svm_classifiers = {}
    for i in range(CLASSES):
        svm_classifiers[i] = {}
        for j in range(i+1,CLASSES):
            xpair,ypair = getDataPairForSVM(data[i],data[j])
            wts,b,loss = mySVM.fit(xpair,ypair,learning_rate=0.00001,maxItr=1000)
            svm_classifiers[i][j] = (wts,b)
            
            plt.plot(loss)
            plt.show()
            
    
    return svm_classifiers


# In[15]:


svm_classifiers = trainSVMs(image_data,labels)


# ### Prediction

# In[16]:


def binaryPredict(x,w,b):
    z  = np.dot(x,w.T) + b
    if z>=0:
        return 1
    else:
        return -1


# In[17]:


def predict(x):
    
    count = np.zeros((CLASSES,))
    
    for i in range(CLASSES):
        for j in range(i+1,CLASSES):
            w,b = svm_classifiers[i][j]
            #Take a majority prediction 
            z = binaryPredict(x,w,b)
            
            if(z==1):
                count[j] += 1
            else:
                count[i] += 1
    
    final_prediction = np.argmax(count)
    #print(count)
    return final_prediction


# In[18]:


# print(predict(image_data[0]))
# print(labels[0])


# In[19]:


def accuracy(x,y):
    
    count = 0
    for i in range(x.shape[0]):
        prediction = predict(x[i])
        if(prediction==y[i]):
            count += 1
            
    return count/x.shape[0]


# In[20]:


accuracy(image_data,labels)


# ### SVM using sklearn

# In[21]:


from sklearn import svm


# In[22]:


svm_classifier = svm.SVC(kernel='linear',C=1.0)


# In[23]:


svm_classifier.fit(image_data,labels)
svm_classifier.score(image_data,labels)
