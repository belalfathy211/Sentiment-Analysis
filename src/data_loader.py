import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split


class IMDBDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.tokens = data["tokens"]
        self.labels = torch.tensor(data["labels"], dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx], dtype=torch.long), self.labels[idx]


class SmartCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        sequences, labels = zip(*batch)
        padded_sequences = pad_sequence(
            sequences, batch_first=True, padding_value=self.pad_idx
        )
        attention_mask = (padded_sequences != self.pad_idx).bool()
        labels = torch.stack(labels)
        return padded_sequences, attention_mask, labels


def get_loaders(data_path, batch_size=32, pad_idx=0, train_ratio=0.8):
    full_dataset = IMDBDataset(data_path)
    collate_fn = SmartCollate(pad_idx)
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


if __name__ == "__main__":
    processed_path = (
        "/home/belal/projects/Sentiment-Analysis/data/processed/processed_data.pt"
    )
    train_loader, val_loader = get_loaders(
        data_path=processed_path, batch_size=16, pad_idx=0, train_ratio=0.8
    )
    x, attention_mask, y = next(iter(train_loader))
    
    print(f"Batch Loaded Successfully!")
    print(f"X Shape (Padded): {x.shape}")
    print(f"attention_mask Shape: {attention_mask.shape}")
    print(f"Y Shape: {y.shape}")
