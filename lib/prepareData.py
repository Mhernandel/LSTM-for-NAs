import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class PrepareData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaled_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path, index_col=0)
        self.data = self.data.interpolate().bfill().ffill()
    
    def scale_data(self):
        self.scaled_data = self.scaler.fit_transform(self.data)
        return self.scaled_data
    
    def reshape_data_for_lstm(self):
        num_features = self.data.shape[1]
        num_samples = self.data.shape[0]
        if self.scaled_data is not None:
            return np.reshape(self.scaled_data, (num_samples, num_features, 1))
        else:
            raise ValueError("Data not scaled yet. Please scale the data before reshaping.")
            
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
