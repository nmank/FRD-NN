from torchvision import models, transforms
import torch 

def choose_model(model_name):

    if model_name == 'raw_triangle_nn_3layer':
        model = torch.nn.Sequential(
                                torch.nn.Flatten(),
                                torch.nn.Linear(784, 500),
                                torch.nn.ReLU(),
                                torch.nn.Linear(500, 100),
                                torch.nn.ReLU(),
                                torch.nn.Linear(100, 10),
                                torch.nn.ReLU())
    
    elif model_name == 'frd_triangle_nn_3layer':
        model = torch.nn.Sequential(
                                torch.nn.Linear(728, 500),
                                torch.nn.ReLU(),
                                torch.nn.Linear(500, 100),
                                torch.nn.ReLU(),
                                torch.nn.Linear(100, 10),
                                torch.nn.ReLU())

    elif model_name == 'raw_cnn_1':
        model = torch.nn.Sequential(
                                torch.nn.Conv2d(1, 10, kernel_size=5),
                                torch.nn.Conv2d(10, 20, kernel_size=5),
                                torch.nn.Dropout2d(),
                                torch.nn.Linear(320, 50),
                                torch.nn.ReLU(),
                                torch.nn.Linear(50, 10))

    return model

def choose_transforms(transforms_name):

    if transforms_name == 'raw_nn_RRC':
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ]),
        }    

    elif transforms_name == 'raw_nn':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ]),
        }


    elif transforms_name == 'frd_nn':
        data_transforms = {
            'train': transforms.Compose([
            ]),
            'test': transforms.Compose([
            ]),
        }

    return data_transforms