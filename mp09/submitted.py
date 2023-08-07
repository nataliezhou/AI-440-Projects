# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        self.transform = transform
        self.target_transform = target_transform
        dataset = []
        for file in data_files:
            data = unpickle(file)
            image = np.array(data[b"data"])
            label = data[b"labels"]
             # reshape image to height x width x channel
            image = image.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            for i in range(len(label)):
                dataset.append((image[i], label[i]))
        self.data = dataset
        

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    dataset = CIFAR10(data_files, transform=transform)
    return dataset


"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    dataloader = DataLoader(dataset, batch_size=loader_params["batch_size"], shuffle=loader_params["shuffle"])
    return dataloader


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################
        
        backbone = resnet18(pretrained=True)
        backbone.load_state_dict(torch.load("resnet18.pt"))
        for param in backbone.parameters():
            param.requires_grad = False

        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, 10) 
        for param in backbone.fc.parameters():
            param.requires_grad = True # unfreeze the last fc layer

        self.backbone = backbone
        self.conv = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################

        return self.backbone(x)

        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    model = FinetuneNet()
    if trained:
        model.load_state_dict(torch.load("finetune_model.pt"))
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    return model


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    if optim_type == "Adam":
        optimizer = torch.optim.Adam(model_params, lr=hparams["lr"])
    elif optim_type == "SGD":
        optimizer = torch.optim.SGD(model_params, lr=hparams["lr"])
    else:
        return None

    return optimizer


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################

    for _, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """

    test_acc = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print("Test acc:", test_acc)

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    
    print("here")
    batch_size = 32
    num_epochs = 3
    learning_rate = 0.001
    
    train_dataset = build_dataset(data_files=["cifar10_batches/data_batch_1"], transform=get_preprocess_transform('train'))
    test_dataset = build_dataset(data_files=["cifar10_batches/test_batch"], transform=get_preprocess_transform('test'))

    train_dataloader = build_dataloader(train_dataset, {"batch_size": batch_size, "shuffle": True})
    test_dataloader = build_dataloader(test_dataset, {"batch_size": batch_size, "shuffle": False})
    model = build_model()
    print("here1")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer("SGD", model.parameters(), {"lr": learning_rate})
    for i in range(num_epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
        
    return model
    
