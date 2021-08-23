from matplotlib import pyplot as plt 
import torch
import numpy as np 
import torchvision
from PIL import Image
from FRD_Functions import circle_values
from scipy import fft



radius = 6
angles = [0,30]
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']

maxR = 13
center = [13,13]
numPoints = 8


mnist_train_dataset = torchvision.datasets.MNIST('./mnist_dataset_torch', train = True)

c_vals = {}
fft_vals = {}

only_one = True
for d,l in mnist_train_dataset:
    if only_one:
        for angle in angles:
            image = d.rotate(angle = angle, center = center)

            c_vals[str(angle)], x_vals, y_vals = circle_values(center, radius, numPoints*radius, np.array(image))

            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.title('Rotated '+str(angle)+' Degrees')
            plt.axis('off')
            plt.scatter(13,13, color = 'red', marker = 'x', s = 3)
            plt.scatter(x_vals, y_vals, color = 'red', s = 3)
            plt.savefig('./plots/raw_image'+str(angle)+'.png')

            fft_vals[str(angle)] = abs(fft(c_vals[str(angle)]))[:numPoints*(radius)+1]
    only_one = False


i=0
for angle in angles:
    plt.figure('interp')
    plt.plot(c_vals[str(angle)], color = colors[i], linestyle = linestyles[i], label = str(angle)+' degrees')

    plt.figure('fft')
    plt.plot(fft_vals[str(angle)], color = colors[i], linestyle = linestyles[i], label = str(angle)+' degrees')

    i+=1

plt.figure('interp')
plt.xlabel('Point Number')
plt.ylabel('Interpolated Image Value')
plt.legend()
plt.savefig('compare_interp.png')

plt.figure('fft')
plt.xlabel('Point Number')
plt.ylabel('Magnitude of Fourier Transform')
plt.legend()
plt.savefig('compare_fft.png')

plt.figure()
plt.plot(c_vals[str(angles[0])], label = str(angle)+' degrees')
plt.xlabel('Point Number')
plt.ylabel('Interpolated Image Value')
plt.savefig('./plots/interpolation.png')

plt.figure()
plt.plot(fft_vals[str(angles[0])], label = str(angle)+' degrees')
plt.xlabel('Point Number')
plt.ylabel('Magnitude of Fourier Transform')
plt.savefig('./plots/fft.png')


