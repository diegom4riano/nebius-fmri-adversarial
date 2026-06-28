import torch
import numpy as np
from torch.utils.data import Dataset
MAX_SENTENCE_LENGTH = 18000


class ECGDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list, target_list):
        """
        @param data_list: list of newsgroup tokens
        @param target_list: list of newsgroup targets

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """

        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]


def ecg_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    B = len(batch)
    data_array  = np.zeros((B, MAX_SENTENCE_LENGTH), dtype=np.float32)
    label_list  = []
    length_list = []

    for i, datum in enumerate(batch):
        length_list.append(datum[1])
        label_list.append(datum[2])
        remainder = MAX_SENTENCE_LENGTH - datum[1]
        pad_left  = int(remainder / 2)
        pad_right = remainder - pad_left
        data_array[i, pad_left : pad_left + datum[1]] = datum[0]

    return [torch.from_numpy(data_array).unsqueeze(-2),
            torch.LongTensor(length_list),
            torch.LongTensor(label_list)]