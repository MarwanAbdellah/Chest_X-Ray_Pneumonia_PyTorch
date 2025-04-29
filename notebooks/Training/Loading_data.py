#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().run_line_magic('cd', '..')


# # Importing Libraries needed

# In[16]:


import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


# # Importing our DataFrames

# In[17]:


df = pd.read_csv('E:\Github\Pneumonia\Chest_X-Ray_Pneumonia_PyTorch\df_resized_aug.csv')
df.head()


# # Creating the Objects needed to create the loader of our data

# In[18]:


class XrayDataset(Dataset):
    def __init__(self, df_split, transform):
        self.df_split = df_split
        self.transform = transform

    def __len__(self):
        return len(self.df_split)

    def __getitem__(self, index):
        img_path = self.df_split.iloc[index]['resized_path']
        clas = self.df_split.iloc[index]['clas']
        # Load the image and convert it to grayscale
        image = Image.open(img_path).convert("L")

        img_trans = self.transform(image)

        return img_trans, clas


# # Implementing our data into our loaders

# ### Training DataLoader

# In[19]:


train_dataset = XrayDataset(df_split=df[df['split'] == 'train'], transform = transforms.ToTensor())

# Create a DataLoader for the training dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# #### Visualizing samples from our train loader

# In[20]:


# Get one batch
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Plot 6 images
plt.figure(figsize=(12, 6))
for i in range(6):
    img = images[i]
    img = img.permute(1, 2, 0).numpy() 
    plt.subplot(2, 3, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Label: {labels[i].item()}")
    plt.axis('on')
plt.tight_layout()
plt.show()


# ### Validation DataLoader

# In[21]:


val_dataset = XrayDataset(df_split=df[df['split'] == 'val'], transform=transforms.transforms.ToTensor())
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


# ### Test DataLoader

# In[22]:


test_dataset = XrayDataset(df_split=df[df['split'] == 'test'], transform=transforms.transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

