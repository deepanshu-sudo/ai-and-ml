"""
The provided Python code defines a Convolutional Neural Network (CNN) class using PyTorch, a popular machine learning library. This class is a subclass of nn.Module, which is the base class for all neural network modules in PyTorch.

The __init__ method is the constructor for the CNN class. It takes two optional parameters: num_channels (default is 3, corresponding to RGB channels of an image) and num_classes (default is 12, the number of output classes). The super(CNN, self).__init__() line is calling the constructor of the superclass nn.Module.

The CNN class has two convolutional layers (conv1 and conv2), each followed by a ReLU activation function (relu1 and relu2) and a max pooling layer (pool1 and pool2). The convolutional layers are used to extract features from the input images, the ReLU activation function introduces non-linearity into the model, and the max pooling layers reduce the spatial dimensions of the output from the convolutional layers.

After the convolutional and pooling layers, the output is flattened and passed through two fully connected layers (fc1 and fc2). The first fully connected layer has 39200 input features and 128 output features, and the second fully connected layer has 128 input features and num_classes output features. The ReLU activation function (relu3) is applied after the first fully connected layer.

The logsoftmax layer applies the log softmax function to the output of the second fully connected layer. This is useful for multiclass classification problems, as it gives the log probabilities of each class.

The forward method defines the forward pass of the network. It takes an input tensor x and passes it through each layer of the network in order. The output of the network is the output of the logsoftmax layer.
"""

from torch import nn
from torch import flatten

class CNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=12):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=num_channels,out_channels=64, kernel_size=(3,3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.fc1 = nn.Linear(in_features=39200,out_features=128)
        
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
        self.logsoftmax = nn.LogSoftmax(dim=1) # comment for loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = flatten(x,1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        out = self.logsoftmax(x)
        return out
