import os
import shutil
import random

'''
Does a 60/40 train/test split of the FRDs in a new directory.

the resulting directory structure:

new_base_dir
    0degrees
        train
            0
                sample0.pt
                .
                .
                .
            1
            2
            .
            .
            .
            9
        test
    10degrees
    20degrees
    .
    .
    .
    350degrees
    all_train_test_split
        train
            0
                sample0_0.pt
                .
                .
                .
            1
            2
            .
            .
            .
            9
        test

'''

#directory where FRDs are coming from 
base_dir = '/data4/mankovic/FRD-NN/FRD_datasets/mnist/' 
#directory where the train/test split will be
new_base_dir = '/data4/mankovic/FRD-NN/FRD_datasets/mnist_splits/'

os.mkdir(new_base_dir)



#make a train test split for each angle of rotation
for d in range(0,360,10):
    degrees_dir = str(d)+'degrees'
    os.mkdir(new_base_dir+degrees_dir)
    os.mkdir(new_base_dir+degrees_dir+'/train')
    os.mkdir(new_base_dir+degrees_dir+'/test')

    for num in range(10):
        os.mkdir(new_base_dir+degrees_dir+'/train/'+str(num))
        os.mkdir(new_base_dir+degrees_dir+'/test/'+str(num))

        current_dir = base_dir + degrees_dir + '/' + str(num)

        for f in os.listdir(current_dir):
            current_file = current_dir + '/' + f
            if os.path.isfile('./raw_datasets/mnist_splits/0degrees/train/'+str(num)+'/'+f[:-3]+'.png'):
                shutil.copyfile(current_file, new_base_dir+degrees_dir+'/train/'+str(num)+'/'+f)
            else:
                shutil.copyfile(current_file, new_base_dir+degrees_dir+'/test/'+str(num)+'/'+f)


#make the all_train_test_split folder with a train and test split with all the angles of rotations
os.mkdir(new_base_dir+'all_train_test_split')
os.mkdir(new_base_dir+'all_train_test_split/train/')
os.mkdir(new_base_dir+'all_train_test_split/test/')

for d in range(0, 360, 10):
    degrees_dir = str(d)+'degrees'
    for num in range(10):
        train_dir = new_base_dir+degrees_dir+'/train/'+str(num)
        for f in os.listdir(train_dir):
            shutil.copyfile(train_dir+'/'+f, new_base_dir+'all_train_test_split/train/'+str(num)+'/'+f[:-3]+'_'+str(d)+'.pt')
            
        test_dir = new_base_dir+degrees_dir+'/test/'+str(num)
        for f in os.listdir(test_dir):
            shutil.copyfile(test_dir+'/'+f, new_base_dir+'all_train_test_split/test/'+str(num)+'/'+f[:-3]+'_'+str(d)+'.pt')