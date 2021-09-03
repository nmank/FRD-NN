import train_model as tm 
from torchvision import models, transforms
from torch import optim
import torch 
import os
import my_models as mm
'''
This script is the workhorse. This trains a NN on FRDs or raw images and 
saves the models and the train/test loss and accuracy at each epoch.


'''


# directory of the train/test split
datapath = '/data4/mankovic/FRD-NN/raw_datasets/mnist_splits/all_train_test_split/'

# directory for saving outputs
base_dir = '/data4/mankovic/FRD-NN/experiments/mnist/raw/rotation_augmentation/raw_triangle_nn_3layer_lr_p01/'

# make output directories
os.mkdir(base_dir)
os.mkdir(base_dir + 'model')

save_model_location = base_dir +'model/model'
save_csv_location = base_dir +'training_stats.csv'


# batch size for the 
batch_size=64 

# the transform to use on train/test data
data_transforms = mm.choose_transforms('raw_nn_RRC')

# the model to be trained
model = mm.choose_model('raw_triangle_nn_3layer')

# the optimizer
opt = optim.Adadelta(model.parameters(), lr=.01)

#train the model
tm.train_model(datapath, model, data_transforms, opt, save_model_location, save_csv_location, 80, batch_size, frd_normalize = False)

