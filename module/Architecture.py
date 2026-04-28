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
        # 3 hidden layers
        self.fcn1 = nn.Linear(input, hidden_size[0])
        self.fcn2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fcn3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fcn4 = nn.Linear(hidden_size[2], output) 
                
    def forward(self, x):
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = F.relu(self.fcn3(x))
        return F.log_softmax(self.fcn4(x), dim=1)
    

class CNN(Architecture):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 47) 
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
        
