import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic

class SensorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        
        # load raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

        # Identify the feature columns dynamically
        self.feature_columns = [col for col in self.df.columns if col.startswith('feature_')]
        self.input_columns = ['x', 'y', 'w', 'h'] + self.feature_columns
        self.target_columns = ['x', 'y', 'w', 'h']

    def __len__(self):
        # return number of sensors
        return len(self.df.groupby(by=["reindexed_id"]))
        # return len(self.df.groupby(by=["reindexed_id

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        # Sensors are indexed from 1
        idx = idx + 1

        # Get the data for the given reindexed_id
        data = self.df[self.df["reindexed_id"] == idx]
        data_length = len(data)
        frames = data["frame"].values
        
        # Ensure there's enough data for the training and forecast windows
        # if data_length <= self.T + self.S:
        #     raise ValueError(f"Not enough data for reindexed_id {idx}: data_length {data_length}, T {self.T}, S {self.S}")

        # Randomly select a starting point for the training window
        start = 0
       
        # Get the sensor number (file name)
        sensor_number = str(data[["File_Name"]].iloc[start].values.item())
        
        # Create indices for input and target windows
        index_in = torch.tensor([i for i in range(start, start + self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        
        # Extract input and target sequences
        input_feat = torch.tensor(data[self.input_columns][start : start + len(data)].values, dtype=torch.float32)
        # bbox = 0
        # true = torch.tensor(data[self.target_columns][start + self.T : start + self.T + self.S].values, dtype=torch.float32)

        return input_feat,  sensor_number, frames
    
if __name__ == '__main__':
    # Assuming 'SensorDataset' class and 'test_csv' are defined
    test_csv = 'test_dataset.csv'
    training_length = 12
    forecast_window = 12
    test_dataset = SensorDataset(csv_name=test_csv, root_dir="Data/", training_length=training_length, forecast_window=forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    print(len(test_dataloader))

        # Example iteration through the dataloader
    for input_feat, sensor_number, frames in test_dataloader:
        print("input_feat:", input_feat.shape)
        bbox= input_feat[:,:,:4]
        print("bbox:", tt)
        print("Sensor Number:", sensor_number)
        print('=========')

