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


def train_model(datapath, model, transformations, opt, save_model_location, save_csv_location, num_epochs = 70, batch_size = 64):

    if 'FRD' in datapath:

        def my_loader(path):
            return torch.load(path)

        image_dataset = {x: datasets.DatasetFolder(os.path.join(datapath, x), loader = my_loader, transform = transformations[x], extensions = ['.pt']) for x in ['train', 'test']}
    else:
        image_dataset = {x: datasets.ImageFolder(os.path.join(datapath, x), transformations[x]) for x in ['train', 'test']}
        
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=batch_size, shuffle=x=='train', num_workers=4) for x in ['train', 'test']}
    num_classes=len(image_dataset['train'].classes)

    criterion = torch.nn.CrossEntropyLoss()

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
                # inputs = inputs.to(device)
                # labels = labels.to(device)

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
                
        print()
        torch.save(model, save_model_location+str(epoch)+'.pth')
        

    plt.figure()
    plt.plot()


    training_stats = pandas.DataFrame(columns = ['Train Loss', 'Test Loss', 'Train Accuracy', 'Test Accuracy','Average Train Accuracy', 'Average Test Accuracy'])
    training_stats['Train Loss'] = [l for l in train_loss]
    training_stats['Test Loss'] = [l for l in val_loss]
    training_stats['Train Accuracy'] = [l.cpu().detach().item() for l in train_acc]
    training_stats['Test Accuracy'] = [l.cpu().detach().item() for l in val_acc]
    training_stats['Average Train Accuracy'] = [l for l in train_avgacc]
    training_stats['Average Test Accuracy'] = [l for l in val_avgacc]
    training_stats.to_csv(save_csv_location)