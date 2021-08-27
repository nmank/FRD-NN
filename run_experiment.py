import train_model as tm 
from torchvision import models, transforms
from torch import optim
import torch 
import os
import my_models as mm


datapath = './FRD_datasets/mnist_splits/0degrees/'

base_dir = './experiments/mnist/frd/frd_triangle_nn_3layer_norml_lr_p001/'

# os.mkdir(base_dir)
# os.mkdir(base_dir + 'model')

save_model_location = base_dir +'model/model'
save_csv_location = base_dir +'training_stats.csv'



batch_size=64


data_transforms = mm.choose_transforms('frd_nn')

model = mm.choose_model('frd_triangle_nn_3layer')

opt = optim.Adadelta(model.parameters(), lr=.001)

# nSamples = list(dict(Counter(image_dataset['train'].targets)).values())
# weights = torch.tensor([1 - (x / sum(nSamples)) for x in nSamples]).to(device)
tm.train_model(datapath, model, data_transforms, opt, save_model_location, save_csv_location, 250, batch_size)
