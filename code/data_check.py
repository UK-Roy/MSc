import numpy as np
import pandas as pd

from network import SafetyModel
from dataset import RobotSafetyDataset

from torch.utils.data import Dataset, DataLoader

# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import json

import tqdm

# print("Shape of feature vector X:", state_action.shape)
# print(f"label: {label[:10]}")

# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

num_epochs = 10  # size of number of epoch
batch_size = 4  # size of each batch

# Instantiate the dataset
dataset = RobotSafetyDataset(root_dir=f'C:\\Users\\tanveer\\thesis\\safety-gymnasium-main\\SafetyCarGoalTrainDataset')

# Create a DataLoader for batching
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch_sequences, batch_label in dataloader:
        print(f"Batch of Sequence: {batch_sequences}\nBatch of Label: {batch_label}")
        
        

