from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import librosa  # Import librosa
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Augmentation functions
def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def pitch_shift(waveform, sample_rate, n_steps=2):
    waveform_np = waveform.squeeze().numpy()
    y_shifted = librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=n_steps)
    return torch.tensor(y_shifted).unsqueeze(0)

def preprocess_audio(audio_path, augmentation=None):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=feature_extractor.sampling_rate)
        waveform = resampler(waveform)
        sample_rate = feature_extractor.sampling_rate  # Update sample rate after resampling

    if augmentation == "noise":
        waveform = add_noise(waveform)
    elif augmentation == "pitch_shift":
        waveform = pitch_shift(waveform, sample_rate)

    # Pass the sampling_rate argument
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs.input_values.squeeze(0)

# Define EmotionDataset class
class EmotionDataset(Dataset):
    def __init__(self, dataframe, emotion_to_idx, augment=False):
        self.dataframe = dataframe
        self.emotion_to_idx = emotion_to_idx
        self.augment = augment

        # Expand dataset
        self.expanded_data = []
        for _, row in dataframe.iterrows():
            audio_path = row['Path']
            label = row['Emotions']

            # Original sample
            self.expanded_data.append((audio_path, label, None))  # None for no augmentation

            if augment:
                # Augmented samples
                self.expanded_data.append((audio_path, label, "noise"))
                self.expanded_data.append((audio_path, label, "pitch_shift"))

    def __len__(self):
        return len(self.expanded_data)

    def __getitem__(self, idx):
        audio_path, label, augmentation = self.expanded_data[idx]
        audio_data = preprocess_audio(audio_path, augmentation)
        label = self.emotion_to_idx[label]

        return {"input_values": audio_data, "labels": torch.tensor(label)}

    def save_features(self, save_path):
        features = {"input_values": [], "labels": []}

        # Use tqdm for the progress bar
        for i in tqdm(range(len(self)), desc="Saving features"):
            sample = self[i]
            features["input_values"].append(sample["input_values"])
            features["labels"].append(sample["labels"])

        with open(save_path, 'wb') as f:
            pickle.dump(features, f)


# Split the dataset into training and test sets
train_df, test_df = train_test_split(Crema_df, test_size=0.2, random_state=42)

# Create the datasets
train_dataset = EmotionDataset(train_df, emotion_to_idx, augment=True)
test_dataset = EmotionDataset(test_df, emotion_to_idx, augment=False)

# Save the features for training and testing separately
train_dataset.save_features("/content/drive/My Drive/features_train.pkl")
test_dataset.save_features("/content/drive/My Drive/features_test.pkl")

print(f"Training features saved to /content/drive/My Drive/features_train.pkl")
print(f"Test features saved to /content/drive/My Drive/features_test.pkl")
