from matplotlib import pyplot as plt 
import torch
import numpy as np 
import torchvision
from PIL import Image
import FRD_Functions
from scipy import fft

'''
Plots some intermediate steps of FRD generation. Saves all the plots as files in ./plots.


TODO: 
    -clean up redundant inputs
    -more commenting
'''

########
#Inputs
radius = 6
angles = [20,40] #angles for rotational invariance plots


linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']

maxR = 13 #max FRD radius
center = [13,13] #center for FRD generation
numPoints = 8 #number of points for the smallest ring
########


mnist_train_dataset = torchvision.datasets.MNIST('/data4/mankovic/FRD-NN/mnist_dataset_torch', train = True)


c_vals = {}
fft_vals = {}

only_one = True
for d,l in mnist_train_dataset:
    if only_one:
        for angle in angles:
            image = d.rotate(angle = angle, center = center)

            if angle == 0:
                frd = FRD_Functions.FRDs(np.array(image), numPoints, maxR, center)
                plt.figure()
                plt.plot(frd)
                plt.title('Fourier Ring Descriptor')
                plt.ylabel('Magnitude of Fourier Coefficient')
                plt.savefig('./plots/whole_frd_0degree.png')
                plt.close()

                plt.figure()
                plt.imshow(torchvision.transforms.RandomResizedCrop((28,28))(image),cmap = 'gray')
                plt.axis('off')
                plt.title('RandomResizeCrop')
                plt.savefig('./plots/raw_image_RRC.png')
                plt.close()

            c_vals[str(angle)], x_vals, y_vals = FRD_Functions.circle_values(center, radius, numPoints*radius, np.array(image))

            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.title('Rotated '+str(angle)+' Degrees')
            plt.axis('off')
            plt.savefig('./plots/raw_image'+str(angle)+'.png')
            plt.scatter(13,13, color = 'red', marker = 'x', s = 3)
            plt.scatter(x_vals, y_vals, color = 'red', s = 3)
            plt.savefig('./plots/raw_image'+str(angle)+'dots.png')
            plt.close()

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
plt.savefig('./plots/compare_interp.png')
plt.close()

plt.figure('fft')
plt.xlabel('Point Number')
plt.ylabel('Magnitude of Fourier Transform')
plt.legend()
plt.savefig('./plots/compare_fft.png')
plt.close()

plt.figure()
plt.plot(c_vals[str(angles[0])], label = str(angle)+' degrees')
plt.xlabel('Point Number')
plt.ylabel('Interpolated Image Value')
plt.savefig('./plots/interpolation.png')
plt.close()

plt.figure()
plt.plot(fft_vals[str(angles[0])], label = str(angle)+' degrees')
plt.xlabel('Point Number')
plt.ylabel('Magnitude of Fourier Transform')
plt.savefig('./plots/fft.png')
plt.close()


###########################################################################################

#plots of raw images

########
#More Inputs
radius = 6
angles = [0,50,100,300]
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']

maxR = 13
center = [13,13]
numPoints = 8
########

mnist_train_dataset = torchvision.datasets.MNIST('/data4/mankovic/FRD-NN/mnist_dataset_torch', train = True)

c_vals = {}
fft_vals = {}

only_one = True
for d,l in mnist_train_dataset:
    if only_one:
        for angle in angles:
            image = d.rotate(angle = angle, center = center)

            plt.figure()
            plt.imshow(image, cmap = 'gray')
            plt.title('Rotated '+str(angle)+' Degrees')
            plt.savefig('./plots/raw_image'+str(angle)+'.png')
            plt.close()

            frd = FRD_Functions.FRDs(np.array(image), numPoints, maxR, center)
            plt.figure()
            plt.plot(frd)
            plt.title('Fourier Ring Descriptor')
            plt.ylabel('Magnitude of Fourier Coefficient')
            plt.savefig('./plots/whole_frd_'+str(angle)+'degree.png')
            plt.close()
    only_one = False