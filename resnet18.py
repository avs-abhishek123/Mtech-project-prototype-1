import os
# loads the OS 

import time
#uses time

from PIL import Image
# use to manipulate images

#from pdb import set_trace as bp 
# can be used for putting up a break point while debugging

import matplotlib.pyplot as plt

import numpy as np
# used to manipulate arrays

import torch
# Deep Learning applications using GPUs and CPUs.

from torch.utils import data
from torch import nn
# module to help us in creating and training of the neural network

from torch import optim
# package implementing various optimization algorithms

import torch.nn.functional as F
# Used to apply Convolution functions etc
# https://pytorch.org/docs/stable/nn.functional.html

from torchvision import datasets, transforms, models

import sys


class CustomizedClassificationPyTorchDataset(data.Dataset):
    def __init__(
        self,
        imageFolderRoot,
        img_size=(224,224),
        mean= [0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    ):
        self.img_size = img_size
        self.transforms = transforms.Compose([transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
        
        self.files = []
        self.lbls = []

        #just reading the sub dirs in imageFolderRoot i.e buildings, sea
        # labels = os.listdir(imageFolderRoot)
        labels = ["buildings","sea"]
        
        lbls_indices = dict()

        for index, label in enumerate(labels):
            lbls_indices[label] = index
            
        print("labels :",labels)
        print("lbls_indices dictionary :",lbls_indices)

        for label in labels:    
            images = os.listdir(os.path.join(imageFolderRoot, label))
            for img in images:
                self.files.append(os.path.join(imageFolderRoot,label,img))
                self.lbls.append(lbls_indices[label])
    
    
    #__getitem__() is a magic method in Python, which when used in a class, allows its instances to use the [] (indexer) operators.     
    def __getitem__(self, index):

        img = self.files[index]
        lbl = self.lbls[index]
        img = Image.open(img).convert('RGB') 
        img = img.resize(self.img_size)
        img = self.transforms(img)
        return img, lbl
    
    # used to find the length of the instance attributes
    # When we invoke the len() method the length of the list named item is 
    # returned that is defined inside the __len__ method.
    def __len__(self):
        return len(self.files)


#FOR VS CODE

#transformname=input("enter the transformname : ")
#PyTorch expects the data to be organized by folders with one folder for each class.
#sys.stdout = open("/mc2/SaiAbhishek/Sprint62/STUDCL-9999-Evaluate improve-Model-performance/AccuracyLogs/"+transformname+"Log.txt", 'w+')

# ===========================================================================

# FOR COLAB
sys.stdout = open('AccuracyLogs/sample_accuracy_log/balanced.txt', 'w+')

# for augmented dataset log
# print("===========================================================================")
# print("                               ACCURACY LOG                                ")
# print("===========================================================================")
# print()
# #print("Result for the Augmentation Name :",transformname[3:])[only for augmented datasets]
# print()
# print( "Model used for Training : RESNET18" )
# print()
# print("Dataset Name : Intel Scenery")
# print()
# print("Classes : buildings , sea ")
# print()
# #print("Minority Class upon which augmentation was applied :",transformname[:3]) [only for augmented datasets]
# print()
# print( "Developer Name : Allena Venkata Sai Abhishek")
# print()
# print("===========================================================================")
# print()


# for balanced dataset log

print("===========================================================================")
print("                               ACCURACY LOG                                ")
print("===========================================================================")
print()
print("Result for the Dataset Name : Balanced Dataset")
print()
print( "Model used for Training : RESNET18" )
print()
print("Dataset Name : Intel Scenery")
print()
print("Classes : buildings , sea ")
print()
print( "Developer Name : Allena Venkata Sai Abhishek")
print()
print("===========================================================================")
print()


# For VS CODE
# train datset
#traindatadir = "/mc2/SaiAbhishek/Sprint62/STUDCL-9999-Evaluate improve-Model-performance/AugmentedDataset/"+transformname+"/train"

# fixed test dataset

#testdatadir = "/mc2/SaiAbhishek/Sprint62/STUDCL-9999-Evaluate improve-Model-performance/StaticTestSet/test"

#For COLAB

# train datset
traindatadir = "Datasets/balancedDataset/train"

# fixed test dataset

testdatadir = "Datasets/StaticTestSet/test"


def load_train_test(traindatadir,testdatadir):
    
    print("For train_dataset")
    train_data = CustomizedClassificationPyTorchDataset(traindatadir)        

    print("===============================")

    print("For test_dataset")
    test_data = CustomizedClassificationPyTorchDataset(testdatadir) 
    
    print("===============================")

    num_train = len(train_data)
    print("Total number of images in train dataset :",num_train)
    
    num_test = len(test_data)
    print("Total number of images in test dataset :",num_test)

    trainloader = torch.utils.data.DataLoader(train_data,
                    batch_size=64,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data,
                    batch_size=64,shuffle=True)
    
    return trainloader, testloader


trainloader, testloader = load_train_test(traindatadir,testdatadir)

# using CUDA
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("device used :",device)

  
model = models.resnet18(pretrained=False)
model


for param in model.parameters():
    # Here, the returned param is torch.nn.Parameter class which is a tensor.
    # Since param is a type of tensor, it has shape and requires_grad attributes too.
    
    param.requires_grad = False
    # param.requires_grad is a boolean which tells if the parameter is learnable or not.

# If we want the whole network to change weights, 
# then we should remove that for loop, where we are setting params.requires_grad = False. 
# But its not a good idea as the number of samples are too less compared 
# to the number of weights. so, its advised to freeze all the layers except fc

# we are setting them false for the pretrained weights. 
# For the last fc layer that we created based on the number of classes, 
# the grads are being calculated, so training has no issue.



# The forward() method of Sequential accepts any
# input and forwards it to the first module it contains. It then
# "chains" outputs to inputs sequentially for each subsequent module,
# finally returning the output of the last module.

model.fc = nn.Sequential(nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))

# Using Sequential to create a model. When `model` is run,
# input will first be passed to `Linear(512, 512)`. The output of
# `Linear(512, 512)` will be used as the input to the first
# `ReLU`; the output of the first `ReLU` will become the input
# for `Dropout(0.2)`. the output of the first `Dropout(0.2)` will 
# become the input for `Linear(512, 2)`.Finally, the output of
# `Linear(512, 2)` will be used as input to the second `LogSoftmax`

criterion = nn.NLLLoss()
#  Loss functions define what a good prediction is and isn't.
# a way to measure how well the model is performing.

optimizer = optim.Adam(model.fc.parameters(), lr=0.0003)
# Optimizers are algorithms or methods used to change the attributes of the neural network 
# such as weights and learning rate to reduce the losses.

model.to(device)
# now u can also see the last sequential layer that I have added to the Resnet model
# inorder to use get the probabilities which can be used to calculate the accuracies



since = time.time()

epochs = 10
running_loss = 0.0

train_accuracy_list = []

correct_predictions=0
total_labels = 0

for epoch in range(epochs):

    model.train()

    for inputs, labels in trainloader:
        
        #=============================

        # augmented_imgs=fun(inputs) 
        # augmented_lbls=fun(inputs,labels)

        # inputs.extend(augmented_imgs)
        # lbls.extend(augmented_lbls)

        #=============================

        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients (clearing optimizer)
        optimizer.zero_grad()
        
        logps = model.forward(inputs)

        # logps - log probabilities
        loss = criterion(logps, labels)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()
        #The loss function is the function that computes the distance between the current output of the algorithm and the expected output.
        running_loss += loss.item()

        probs = torch.exp(logps)
        probs=probs.cpu()

        pred_labels = np.argmax(probs.detach().numpy(), axis=1)

        labels=labels.cpu()
        correct_predictions += np.sum(labels.detach().numpy() ==pred_labels)
        total_labels += len(labels)
        epoch_loss=running_loss/len(trainloader)
    Train_accuracy = correct_predictions/total_labels
    train_accuracy_list.append(Train_accuracy)
    print(f" Training Phase : Epoch {epoch+1}/{epochs}..Train loss: {epoch_loss}.. Train accuracy: { Train_accuracy}")

print("The Final Train Loss :",epoch_loss)
print("The Final Train Accuracy :",Train_accuracy)
print("The Best Train Accuracy :",max(train_accuracy_list))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# ============================================== TRAINING DONE =============================================

correct_predictions = 0
model.eval()
total_labels = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        probs = torch.exp(logps)
        probs=probs.cpu()
        pred_labels = np.argmax(probs.detach().numpy(), axis=1)
        labels=labels.cpu()
        correct_predictions += np.sum(labels.detach().numpy() ==pred_labels)
        total_labels += len(labels)

                
Testaccuracy = correct_predictions/total_labels
print("Test Accuracy :",Testaccuracy)


sys.stdout.close()            

# ============================================== TESTING DONE =============================================
