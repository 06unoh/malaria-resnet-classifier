from datasets.malaria_dataset import ImageDataset
from models.custom_resnet import CustomResNet
from utils.train import train
from utils.test import test
from utils.visualize import visualize_prediction
from utils.unzip import unzip
from utils.data_utils import get_mean_std
from utils.transforms import get_train_tf, get_test_tf, get_temp_tf

import zipfile
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


unzip()
filepath='./cell_image'
parasitized=os.listdir('./cell_image/Parasitized/')
uninfected=os.listdir('./cell_image/Uninfected/')
parasitized=[os.path.join('Parasitized',filename) for filename in parasitized]
uninfected=[os.path.join('Uninfected',filename) for filename in uninfected]
allfiles=parasitized+uninfected
labels=[1]*len(parasitized)+[0]*len(uninfected)

temp_tf=get_temp_tf()
temp_ds=ImageDataset(filepath,allfiles,labels,temp_tf)
temp_dl=DataLoader(temp_ds, 32, shuffle=False)

mean, std=get_mean_std(temp_dl)

tf_train=get_train_tf(mean, std)
tf_test=get_test_tf(mean, std)

trainfiles, testfiles, train_label, test_label=train_test_split(allfiles, labels, test_size=0.2, random_state=42, stratify=labels)
trainset=ImageDataset(filepath,trainfiles,train_label,tf_train)
testset=ImageDataset(filepath,testfiles,test_label,tf_test)
trainloader=DataLoader(trainset,32,True)
testloader=DataLoader(testset,32,False)

if __name__=="__main__":    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CustomResNet().to(device)
    criterion=nn.BCEWithLogitsLoss().to(device)
    optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    for epoch in range(40):
        train_loss=train(model, trainloader, criterion, optimizer, device)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')
    test(model, testloader, device)
    visualize_prediction(model, testloader, device, std, mean)