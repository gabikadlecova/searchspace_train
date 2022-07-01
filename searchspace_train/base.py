import pickle
import torch


class TrainedNetwork:
    def __init__(self, hash, checkpoint_path, data_path, data_mode='pickle'):
        self.hash = hash
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.data_mode = data_mode

    def load_data(self):
        if self.data_mode == 'pickle':
            with open(self.data_path, 'rb') as f:
                return pickle.load(f)
        elif self.data_mode == 'torch':
            return torch.load(self.data_path)
        else:
            raise ValueError(f"Unknown data format for loading: {self.data_mode} at {self.data_path}.")

    def load_checkpoint(self, device=None):
        return torch.jit.load(self.checkpoint_path, map_location=device)