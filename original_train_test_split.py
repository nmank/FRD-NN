import os
import shutil
import random

#do a random 60/40 train/test split for the .png images at 0degrees
#we'll copy this for all other train/test splits
#so, don't run this again

base_dir = './raw_datasets/mnist/' 
new_base_dir = './raw_datasets/mnist_splits/'

os.mkdir(new_base_dir)
os.mkdir(new_base_dir+'0degrees')
os.mkdir(new_base_dir+'0degrees/train')
os.mkdir(new_base_dir+'0degrees/test')


degrees_dir = '0degrees'
for num in range(10):
    os.mkdir(new_base_dir+'0degrees/train/'+str(num))
    os.mkdir(new_base_dir+'0degrees/test/'+str(num))
    current_dir = base_dir + degrees_dir + '/' + str(num)
    for f in os.listdir(current_dir):
        current_file = current_dir + '/' + f
        if random.random() <= .6:
            shutil.copyfile(current_file, new_base_dir+'0degrees/train/'+str(num)+'/'+f)
        else:
            shutil.copyfile(current_file, new_base_dir+'0degrees/test/'+str(num)+'/'+f)