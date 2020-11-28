#!/usr/bin/env python
# coding: utf-8

# ### IMAGE CLASSIFICATION USING SVM

# In[1]:


import matplotlib.pyplot as plt
import cv2
import numpy as np


# In[2]:


im=cv2.imread('goku.jpg')               #reads inn bgr format     
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)   #coverts bgr to rgb
original_shape=im.shape
# print(im.shape)


# In[3]:


plt.imshow(im)                             #shows as RGBB Format
plt.show()


# In[4]:


# Flatten each channel of the image
all_pixels=im.reshape((-1,3))
# print(all_pixels.shape)


# In[5]:


from sklearn.cluster import KMeans
dominant_colors = 4 
km=KMeans(n_clusters=dominant_colors)
km.fit(all_pixels)


# In[6]:


centers = km.cluster_centers_
centers = np.array(centers, dtype = 'uint8')
# print(centers)


# ### Plot what all colors are these

# In[7]:


i=1
plt.figure(0, figsize=(4,2))

colors = []
for each_col in centers:
    plt.subplot(1,4,i)
    plt.axis('off')
    i+=1
    
    colors.append(each_col)
    
    #color swatch
    a=np.zeros((100,100,3), dtype='uint8')
    a[:,:,:]=each_col
    plt.imshow(a)
    
plt.show()


# ### Segmenting our original image

# In[8]:


new_img=np.zeros((1385*1628,3),dtype='uint8')
# print(new_img.shape)


# In[9]:


colors


# In[10]:


km.labels_


# In[11]:


for ix in range(new_img.shape[0]):
    new_img[ix]=colors[km.labels_[ix]]
    
new_img= new_img.reshape((original_shape))
plt.imshow(new_img)
plt.show()
