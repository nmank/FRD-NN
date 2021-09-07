import train_model as tm 
from torchvision import models, transforms
from torch import optim
import torch 
import os
import my_models as mm


datapath = '/data4/mankovic/FRD-NN/raw_datasets/mnist_splits/all_train_test_split/'

base_dir = '/data4/mankovic/FRD-NN/experiments/mnist/raw/rotation_augmentation/raw_triangle_nn_3layer_lr_p01/'

# os.mkdir(base_dir)
# os.mkdir(base_dir + 'model')

save_model_location = base_dir +'model/model'
save_csv_location = base_dir +'training_stats.csv'



batch_size=64


data_transforms = mm.choose_transforms('raw_nn_RRC')

model = mm.choose_model('raw_triangle_nn_3layer')

opt = optim.Adadelta(model.parameters(), lr=.01)

# nSamples = list(dict(Counter(image_dataset['train'].targets)).values())
# weights = torch.tensor([1 - (x / sum(nSamples)) for x in nSamples]).to(device)
tm.train_model(datapath, model, data_transforms, opt, save_model_location, save_csv_location, 80, batch_size, frd_normalize = False)

