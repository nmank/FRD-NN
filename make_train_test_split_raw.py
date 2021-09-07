import os
import shutil
import random
'''
Does a 60/40 train/test split of the raw images in a new directory.

the resulting directory structure:

new_base_dir
    0degrees
        train
            0
                sample0.png
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
                sample0_0.png
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

#do pngs


#directory where images are coming from 
base_dir = '/data4/mankovic/FRD-NN/raw_datasets/mnist/' 

#directory where the train/test splits are going
new_base_dir = '/data4/mankovic/FRD-NN/raw_datasets/mnist_splits/'


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
            if os.path.isfile(new_base_dir+'0degrees/train/'+str(num)+'/'+f):
                shutil.copyfile(current_file, new_base_dir+degrees_dir+'/train/'+str(num)+'/'+f)
            else:
                shutil.copyfile(current_file, new_base_dir+degrees_dir+'/test/'+str(num)+'/'+f)

                
                
#make a train test split with all the angles of rotation
os.mkdir(new_base_dir+'all_train_test_split')
os.mkdir(new_base_dir+'all_train_test_split/train/')
os.mkdir(new_base_dir+'all_train_test_split/test/')

for d in range(0, 360, 10):
    degrees_dir = str(d)+'degrees'
    for num in range(10):
        train_dir = new_base_dir+degrees_dir+'/train/'+str(num)
        for f in os.listdir(train_dir):
            shutil.copyfile(train_dir+'/'+f, new_base_dir+'all_train_test_split/train/'+str(num)+'/'+f[:-4]+'_'+str(d)+'.png')
            
        test_dir = new_base_dir+degrees_dir+'/test/'+str(num)
        for f in os.listdir(test_dir):
            shutil.copyfile(test_dir+'/'+f, new_base_dir+'all_train_test_split/test/'+str(num)+'/'+f[:-4]+'_'+str(d)+'.png')