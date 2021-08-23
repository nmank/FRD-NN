from torchvision import datasets, models, transforms
import torch
import glob
import torch.optim as optim
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
from collections import Counter

device='cuda:3'


def val_rotations(model):


    all_preds = []
    for ii in range(0,360,5):
        datapath='/qfs/projects/rain_man/datasets/frd_splits_tim/Rotations_16_RGB/'+str(ii)+'degrees/'

        batch_size=10
        input_size=(224,224)
        #i dont think flips make sense here
        #maybe want color jitter?
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_dataset = {x: datasets.ImageFolder(os.path.join(datapath, x), data_transforms[x]) for x in [ 'test']}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=batch_size,  num_workers=4) for x in [ 'test']}
        num_classes=len(image_dataset['test'].classes)
        

        phase = 'test'
        num_epochs=100
        val_acc=[]
        val_cm=[]
        confmat=np.zeros((num_classes,num_classes))
        running_corrects = 0
        angle_preds = []
        for inputs, labels in dataloaders_dict[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for l,p in zip(labels.cpu().detach().numpy(),preds.cpu().detach().numpy()):
                confmat[l,p]+=1
            running_corrects += torch.sum(preds == labels.data)
            angle_preds.append(outputs.detach())
        epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

        all_preds.append(torch.cat(angle_preds,0))
#         print('angle '+str(ii)+' done!')
    all_preds = torch.stack(all_preds)    

    
#     av_top_preds = np.mean([torch.unique(preds).size() for preds in all_preds.T])
    
#     print(av_top_preds)
    
    return all_preds

datapath='/qfs/projects/rain_man/datasets/frd_splits_tim/Rotations_16_RGB/all_train_split'
batch_size=64

input_size=(224,224)
#i dont think flips make sense here
#maybe want color jitter?
data_transforms = {
    'train': transforms.Compose([
#         transforms.Lambda(crop16),
        transforms.RandomResizedCrop(input_size), #from tim's original code
#         transforms.Resize(input_size),
        transforms.ToTensor(),
#         AddGaussianNoise(0,.01),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
#         transforms.Lambda(crop16),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_dataset = {x: datasets.ImageFolder(os.path.join(datapath, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=batch_size, shuffle=x=='train', num_workers=4) for x in ['train', 'test']}
num_classes=len(image_dataset['train'].classes)
batch_size=64


model=models.resnet18(pretrained=True)
model.fc = torch.nn.Sequential(torch.nn.Linear(512,num_classes),
                               torch.nn.Dropout(p=.5))
model.to(device)

opt = optim.Adadelta(model.parameters(), lr=0.001)

nSamples = list(dict(Counter(image_dataset['train'].targets)).values())
weights = torch.tensor([1 - (x / sum(nSamples)) for x in nSamples]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight = weights)
# scheduler = optim.ExponentialLR(optimizer, gamma=0.9)

num_epochs=70
train_loss=[]
train_acc=[]
train_avgacc=[]
train_cm=[]
val_loss=[]
val_acc=[]
val_avgacc=[]
val_cm=[]

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'test']:
        confmat=np.zeros((num_classes,num_classes))

        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders_dict[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                for l,p in zip(labels.cpu().detach().numpy(),preds.cpu().detach().numpy()):
                    confmat[l,p]+=1
            
                if phase == 'train':
                    loss.backward()
                    opt.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

#         scheduler.step()
        epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)
        epoch_avg_acc = np.mean(np.diag(confmat)/np.sum(confmat,1))

        print('{} Loss: {:.4f} Acc: {:.4f} Avg Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_avg_acc))
        print(confmat.astype(np.int))
        
        if phase == 'train':
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
            train_avgacc.append(epoch_avg_acc)
            train_cm.append(confmat)
        else:
            val_loss.append(epoch_loss)
            val_acc.append(epoch_acc)
            val_avgacc.append(epoch_avg_acc)
            val_cm.append(confmat)
            
            predictions = val_rotations(model)
            
            torch.save(predictions, '/qfs/projects/rain_man/frd_weights/predictions/dropout/resnet18_lr001_weighted_loss_predictions_e'+str(epoch)+'.pt')

    print()
    torch.save(model, '/qfs/projects/rain_man/frd_weights/Big/dropout/resnet18_lr001_weighted_loss_e'+str(epoch)+'.pth')
    

plt.figure()
plt.plot()


training_stats = pandas.DataFrame(columns = ['Train Loss', 'Test Loss', 'Train Accuracy', 'Test Accuracy','Average Train Accuracy', 'Average Test Accuracy'])
training_stats['Train Loss'] = [l for l in train_loss]
training_stats['Test Loss'] = [l for l in val_loss]
training_stats['Train Accuracy'] = [l.cpu().detach().item() for l in train_acc]
training_stats['Test Accuracy'] = [l.cpu().detach().item() for l in val_acc]
training_stats['Average Train Accuracy'] = [l for l in train_avgacc]
training_stats['Average Test Accuracy'] = [l for l in val_avgacc]
training_stats.to_csv('/qfs/projects/rain_man/frd_weights/predictions/dropout/training_stats_resnet18_lr001_weighted_loss.csv')