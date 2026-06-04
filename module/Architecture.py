import torch
import torch.nn as nn
import torch.nn.functional as F

class Architecture(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

class ANN(Architecture):
    def __init__(self, input, hidden_size, output):
        super().__init__()
        self.fcn1 = nn.Linear(input, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.fcn2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])
        self.fcn3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.bn3 = nn.BatchNorm1d(hidden_size[2])
        self.fcn4 = nn.Linear(hidden_size[2], output)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fcn1(x))))
        x = self.drop(F.relu(self.bn2(self.fcn2(x))))
        x = F.relu(self.bn3(self.fcn3(x)))
        return F.log_softmax(self.fcn4(x), dim=1)
    
class CNN(Architecture):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   
        self.conv2 = nn.Conv2d(32, 64, 3, 1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(64 * 5 * 5, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 47)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return F.log_softmax(self.fc3(x), dim=1)
        
