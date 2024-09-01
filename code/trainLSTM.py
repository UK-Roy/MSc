import numpy as np
import pandas as pd
from network import LSTMSafetyNetwork
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import joblib

import torch
import torch.nn as nn
import torch.optim as optim

import tqdm

df = pd.read_csv("C:/Users/tanveer/thesis/safety-gymnasium-main/MSc/code/env.csv")

states = df.drop(['action 0', 'action 1', 'label'], axis='columns', inplace=False).to_numpy()
actions = df[['action 0', 'action 1']].to_numpy()
label = df['label'].to_numpy().astype(int)

state_action = df.drop('label', axis='columns', inplace=False)
lidar_action = state_action.drop([f"state {i}" for i in range(0, 40)], axis='columns', inplace=False).to_numpy()
# Combine states and actions into a single feature array
# X = np.hstack((states, actions))
safety_net = LSTMSafetyNetwork(input_dim=34, hidden_dim=32, num_layers=5)
criterion = nn.BCELoss()
optimizer = optim.Adam(safety_net.parameters(), lr=0.001)

# print("Shape of feature vector X:", state_action.shape)
# print(f"label: {label[:10]}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training at Device: {device}")
# Convert collected data to tensors
safety_net.to(device)
data = torch.tensor(lidar_action, dtype=torch.float32).to(device)
labels = torch.tensor(label, dtype=torch.float32).view(-1, 1).to(device)

# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)
X_train = data[:200, :]
X_test = data[200:, :]

y_train = label[:200, :]
y_test = label[200:, :] 

n_epochs = 200    # number of epochs to run
batch_size = 100  # size of each batch
batches_per_epoch = len(X_train) // batch_size

# Train the network
for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch+1}")
        for i in bar:
            optimizer.zero_grad()
            outputs = safety_net(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            bar.set_postfix(
                loss=float(loss)
            )

y_pred = safety_net(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print(f"End of {epoch+1}, accuracy {acc}")

# Save the trained safety network
torch.save(safety_net.state_dict(), 'lstm_safety_net.pth')

# Load the trained safety network
safety_net.load_state_dict(torch.load('lstm_safety_net.pth'))
