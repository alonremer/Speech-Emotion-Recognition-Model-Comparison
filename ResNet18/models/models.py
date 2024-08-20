import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

class ResNet18ForEmotion(nn.Module):
    def __init__(self, num_classes=6):
        """
        Initializes the ResNet18 model adapted for emotion classification.

        Args:
            num_classes (int): Number of emotion classes. Defaults to 6.
        """
        super(ResNet18ForEmotion, self).__init__()
        
        # Load the pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept single-channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)
        
        # Modify the fully connected layer to output the number of emotion classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output of the model.
        """
        return self.model(x)

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        # Load the pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept single-channel input
        self.resnet.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)  # Dropout after ReLU
        )

        # Replace the fully connected layer with a new one with dropout
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(self.resnet.fc.in_features, 6)
        )

    def forward(self, x):
        return self.resnet(x)
