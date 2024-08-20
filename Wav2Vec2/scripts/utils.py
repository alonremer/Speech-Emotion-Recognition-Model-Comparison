import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer, get_scheduler
import matplotlib.pyplot as plt

# Load features from cache
def load_features(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, "rb") as f:
        features = pickle.load(f)
    return features

# Define the EmotionDataset class to use precomputed features
class EmotionDataset(Dataset):
    def __init__(self, features):
        self.input_values = features['input_values']
        self.labels = features['labels']

    def __len__(self):
        return len(self.input_values)

    def __getitem__(self, idx):
        input_value = self.input_values[idx]
        label = self.labels[idx]
        return {'input_values': torch.tensor(input_value), 'labels': torch.tensor(label)}

# Split dataset into train and validation sets
def split_dataset(dataset, train_size=0.9):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(indices, train_size=train_size, shuffle=True)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# Custom collate function to pad sequences
def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    input_values_padded = pad_sequence(input_values, batch_first=True)
    return {'input_values': input_values_padded, 'labels': labels}
