import torch


class TrainedNetwork:
    def __init__(self, hash, checkpoint_path, data_path):
        self.hash = hash
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path

    def load_data(self):
        return torch.load(self.data_path)

    def load_checkpoint(self, device=None):
        return torch.jit.load(self.checkpoint_path, map_location=device)