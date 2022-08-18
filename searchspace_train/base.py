from typing import Optional, Union

import torch
from abc import abstractmethod


class TrainedNetwork:
    def __init__(self, net_hash: str, checkpoint_path: str, data_path: str):
        self.hash = net_hash
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path

    def load_data(self):
        return torch.load(self.data_path)

    def load_checkpoint(self, device: Optional[str] = None):
        return torch.jit.load(self.checkpoint_path, map_location=device)


class BaseDataset:
    @abstractmethod
    def get_trained_hashes(self) -> List[str]:
        pass

    @abstractmethod
    def get_network(self, net_hash: str, dir_path: Optional[str] = None) -> TrainedNetwork:
        pass


def enumerate_trained_networks(dataset: BaseDataset, with_data: Optional[bool] = False, dir_path: Optional[str] = None,
                               device: Optional[str] = None):
    hash_list = dataset.get_trained_hashes()
    for net_hash in hash_list:
        trained_net = dataset.get_network(net_hash, dir_path)

        checkpoint = trained_net.load_checkpoint(device=device)
        if with_data:
            data = trained_net.load_data()
            yield checkpoint, data
        yield checkpoint
