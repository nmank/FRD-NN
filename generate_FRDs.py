from FRD_Functions import FRDs 
import torch
import numpy as np 
import torchvision
from PIL import Image

'''
Generates raw images and FRDs of MNIST dataset with rotations between 0 and 350 degrees in 10 degree increments,

Note: there is no center pixel of a 28x28 image. so we take the center to be at 13,13 and skip the furthest right column and bottom row of pixels.
'''

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
        image.save('/data4/mankovic/FRD-NN/raw_datasets/mnist/'+str(angle)+'degrees/'+str(l)+'/sample'+str(counts[str(l)])+'.png')
        np_image = np.array(image)
        current_FRD = FRDs(np_image, numPoints, maxR, center)
        torch.save(torch.tensor(current_FRD), '/data4/mankovic/FRD-NN/FRD_datasets/mnist/'+str(angle)+'degrees/'+str(l)+'/sample'+str(counts[str(l)])+'.pt')
    counts[str(l)]+=1


    