import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = 'FALSE'
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
from torch.utils.data import random_split
import sys

sys.path.append('Lab_0/Functions.py')
from Functions import train_model, test_model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print("Using device:", device)

    # Download and prepare CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Specify the data directory
    data_dir = './data'

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_size = int(0.8 * len(trainset))  # 80% for training
    val_size = len(trainset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=2)

    # We load the AlexNet model
    model_Alex_ft = models.alexnet(pretrained=True)

    # Get the number of input features for the final fully connected layer
    num_ftrs = model_Alex_ft.classifier[6].in_features

    # Modify the final fully connected layer to have 10 output classes
    model_Alex_ft.classifier[6] = nn.Linear(num_ftrs, 10)

    # Define the initial learning rate and optimizer
    initial_lr = 0.001
    optimizer_ft = torch.optim.SGD(model_Alex_ft.parameters(), lr=initial_lr, momentum=0.9)

    # Define the loss function
    criterion_ft = nn.CrossEntropyLoss()

    # Define a learning rate scheduler
    scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    epochs = 5
    # Fine-tuning loop with hyperparameter tuning
    train_model(model_Alex_ft, optimizer=optimizer_ft, criterion=criterion_ft, scheduler=scheduler_ft,
                train_loader=trainloader, val_loader=validloader, epochs=epochs)

    test_model(model=model_Alex_ft, criterion=criterion_ft, test_loader=testloader)

    # Feature extraction

    # Modify model for feature extraction
    model_Alex_fe = models.alexnet(pretrained=True)

    # this loop will freeze all layers
    for param in model_Alex_fe.parameters():
        param.requires_grad = False

    # Add custom classifier
    num_classes = 10  # Example: 10 classes
    model_Alex_fe.classifier.add_module('6', nn.Linear(4096, num_classes))

    # Define loss function and optimizer
    criterion_fe = nn.CrossEntropyLoss()
    optimizer_fe = optim.Adam(model_Alex_fe.classifier.parameters(), lr=0.001)
    scheduler_fe = torch.optim.lr_scheduler.StepLR(optimizer_fe, step_size=7, gamma=0.1)

    # Train on extracted features
    epochs = 5
    train_model(model_Alex_fe, optimizer=optimizer_fe, criterion=criterion_fe, scheduler=scheduler_fe,
                train_loader=trainloader, val_loader=validloader, epochs=epochs)

    test_model(model=model_Alex_fe, criterion=criterion_fe, test_loader=testloader)

'''
    The main difference between feature extraction and transfer learning is tha in transfer learning we take 
    the whole model but in feature extraction we train based on specific layer we want mostly the last layer of the model
    meaning we take the weight of the last layer and we tune those. One of the reasons that the two models might differ can 
    be because of the input. those feature of the last layers were obtained by a training of a specific input 
    dataset and the two datasets are very different the results might differ cause in the first case the whole feature are optimized 
    while in feature extraction only the targeted layer feature are optimized instead:
'''

