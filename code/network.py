import torch
import torch.nn as nn
import torch.optim as optim

class SafetyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SafetyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class LSTMSafetyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=5):
        super(LSTMSafetyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        out = lstm_out[:, -1, :]
        # Fully connected layer and sigmoid activation
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

