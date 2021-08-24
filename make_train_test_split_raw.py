import os
import shutil
import random

#do pngs

base_dir = './raw_datasets/mnist/' 
new_base_dir = './raw_datasets/mnist_splits/'

# os.mkdir(new_base_dir)




# for d in range(10,360,10):
#     degrees_dir = str(d)+'degrees'
#     os.mkdir(new_base_dir+degrees_dir)
#     os.mkdir(new_base_dir+degrees_dir+'/train')
#     os.mkdir(new_base_dir+degrees_dir+'/test')

#     for num in range(10):
#         os.mkdir(new_base_dir+degrees_dir+'/train/'+str(num))
#         os.mkdir(new_base_dir+degrees_dir+'/test/'+str(num))

#         current_dir = base_dir + degrees_dir + '/' + str(num)

#         for f in os.listdir(current_dir):
#             current_file = current_dir + '/' + f
#             if os.path.isfile(new_base_dir+'0degrees/train/'+str(num)+'/'+f):
#                 shutil.copyfile(current_file, new_base_dir+degrees_dir+'/train/'+str(num)+'/'+f)
#             else:
#                 shutil.copyfile(current_file, new_base_dir+degrees_dir+'/test/'+str(num)+'/'+f)


# os.mkdir(new_base_dir+'all_train_test_split')
# os.mkdir(new_base_dir+'all_train_test_split/train/')
# os.mkdir(new_base_dir+'all_train_test_split/test/')

for d in range(0, 360, 10):
    degrees_dir = str(d)+'degrees'
    for num in range(10):
        train_dir = new_base_dir+degrees_dir+'/train/'+str(num)
        for f in os.listdir(train_dir):
            shutil.copyfile(train_dir+'/'+f, new_base_dir+'all_train_test_split/train/'+str(num)+'/'+f[:-4]+'_'+str(d)+'.png')
            
        test_dir = new_base_dir+degrees_dir+'/test/'+str(num)
        for f in os.listdir(test_dir):
            shutil.copyfile(test_dir+'/'+f, new_base_dir+'all_train_test_split/test/'+str(num)+'/'+f[:-4]+'_'+str(d)+'.png')