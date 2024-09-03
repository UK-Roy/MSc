import os
import json
import numpy as np

from natsort import natsorted

import torch
from torch.utils.data import Dataset

class RobotSafetyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sequence_dirs = [os.path.join(root_dir, seq_dir) for seq_dir in os.listdir(root_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence_dir = self.sequence_dirs[idx]
        sequence_data, labels = self.load_sequence(sequence_dir)

        if self.transform:
            sequence_data = self.transform(sequence_data)

        return sequence_data, labels

    def load_sequence(self, sequence_dir):
        """
        Load the sequence from a directory. Each sequence is stored in JSON files or
        other formats. This method should return a tensor of shape (sequence_length, num_features).
        """
        sequence_files = natsorted([os.path.join(sequence_dir, f, 'state.json') for f in os.listdir(sequence_dir)])
        sequence = []
        labels = []

        for file in sequence_files:
            with open(file, 'r') as f:
                data_point = json.load(f)
                features, label = self.extract_features(data_point)
                sequence.append(features)
                labels.append(label)

        # Convert the sequence to a NumPy array
        sequence = np.array(sequence)
        labels = np.array(labels)

        # Convert the sequence to a tensor
        sequence = torch.tensor(sequence, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return sequence, labels

    def extract_features(self, data_point):
        """
        Extract features from a single data point. This needs to be customized
        based on the format of your data.
        """
        features = np.concatenate([
            np.array(data_point['accelerometer']),
            np.array(data_point['velocimeter']),
            np.array(data_point['gyro']),
            np.array(data_point['magnetometer']),
            np.array(data_point['ballangvel_rear']),
            np.array(data_point['ballquat_rear']),
            np.array(data_point['goal_lidar']),
            np.array(data_point['hazards_lidar']),
            np.array(data_point['vases_lidar']),
            np.array(data_point['action'])
        ])

        # The label could be cost_sum or a binary label indicating safety (0 for unsafe, 1 for safe)
        # label = np.array([0]) if data_point['cost_sum'] > 0 else np.array([1])  # This is a binary label example
        label = np.array([0]) if data_point['cost_sum'] > 0 else np.array([1])  # This is a binary label example

        return features, label