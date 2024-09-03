import torch
import torch.nn as nn
import torch.optim as optim

class SafetyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(SafetyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Model instantiation
# input_size = len(features)  # This is the size of your input vector
# model = SafetyModel(input_size=input_size, hidden_size=64, output_size=1)

class LSTMSafetyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMSafetyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        out = self.sigmoid(out)
        return out