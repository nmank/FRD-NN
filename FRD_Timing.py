import time
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

stop_it = False
for d,l in mnist_train_dataset:
    if not stop_it:
        start = time.time()
        for angle in range(0,360,10):
            image = d.rotate(angle = angle, center = center)
            # image.save('./raw_datasets/mnist/'+str(angle)+'degrees/'+str(l)+'/sample'+str(counts[str(l)])+'.png')
            np_image = np.array(image)
            current_FRD = FRDs(np_image, numPoints, maxR, center)
            # torch.save(torch.tensor(current_FRD), './FRD_datasets/mnist/'+str(angle)+'degrees/'+str(l)+'/sample'+str(counts[str(l)])+'.pt')
        counts[str(l)]+=1
        total_time = time.time()-start
        print('it took '+str(total_time)+' seconds')
    stop_it = True

print('So it will take ' +str(total_time*60000/60/60)+ ' hours to do it')

    