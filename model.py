from torch import nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Input is torch.Size([1, 240, 256, 3])
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # Output is torch.Size([1, 118, 126, 16])
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # Output is torch.Size([1, 57, 61, 32])
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # Output is torch.Size([1, 27, 29, 32])
        self.fc1 = nn.Linear(27 * 29 * 32, 256)
        self.fc2 = nn.Linear(256, n_actions)

        


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 27 * 29 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class linear_DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(linear_DQN, self).__init__()
        # Assuming input_size = 22 for the tensor of 22 values
        self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)          # Second fully connected layer
        self.fc3 = nn.Linear(64, n_actions)    # Output layer with n_actions

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function after first layer
        x = F.relu(self.fc2(x))  # Activation function after second layer
        x = self.fc3(x) 
        return x
