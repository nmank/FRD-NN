from FRD_Functions import FRDs 
import torch
import numpy as np 
import torchvision
from PIL import Image

maxR = 13
center = [13,13]
numPoints = 8


mnist_train_dataset = torchvision.datasets.MNIST('./mnist_dataset_torch', train = True)

counts = {}
for i in range(10): 
    counts[str(i)] = 0 


for d,l in mnist_train_dataset:
    for angle in range(0,360,10):
        image = d.rotate(angle = angle, center = center)
        image.save('./raw_datasets/mnist/'+str(angle)+'degrees/'+str(l)+'/sample'+str(counts[str(l)])+'.png')
        np_image = np.array(image)
        current_FRD = FRDs(np_image, numPoints, maxR, center)
        torch.save(torch.tensor(current_FRD), './FRD_datasets/mnist/'+str(angle)+'degrees/'+str(l)+'/sample'+str(counts[str(l)])+'.pt')
    counts[str(l)]+=1


    