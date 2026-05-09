import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


class Partition:
    def __init__(self, data, batch_size=10, max_length=256, stride=128, shuffle=True, drop_last=True, world_size=2):
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.world_size = world_size
        self.partitions = {}

    def create_loader(self):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(self.data, tokenizer, self.max_length, self.stride)

        total_size = len(dataset)
        per_partition_size = total_size // self.world_size

        for idx in range(self.world_size):
            start_idx = idx * per_partition_size
            end_idx = start_idx + per_partition_size if idx < self.world_size - 1 else total_size
            partition_indices = list(range(start_idx, end_idx))
            partition_dataset = Subset(dataset, partition_indices)
            self.partitions[idx] = DataLoader(partition_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

    def get_partition(self, rank):
        if not self.partitions:
            return "bruh, run create loader first"
        
        return self.partitions[rank] 