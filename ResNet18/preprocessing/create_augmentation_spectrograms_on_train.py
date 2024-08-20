import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import pandas as pd
from tqdm import tqdm

# Augmentation functions
def add_noise(waveform, noise_level=0.01):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def change_vol(waveform):
    tranform = T.Vol(gain=5.0)
    return tranform(waveform)

def time_masking(waveform, mask_length=2000):
    n_samples = waveform.shape[1]
    if mask_length >= n_samples:
        raise ValueError("Mask length must be smaller than the number of samples in the waveform")
    start = torch.randint(0, n_samples - mask_length, (1,)).item()
    waveform[:, start:start + mask_length] = 0.0
    return waveform

# Spectrogram function
def spectrogram(waveform, sample_rate):
    mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=128)(waveform)
    mel_spectrogram_db = AmplitudeToDB()(mel_spectrogram)
    return mel_spectrogram_db

# List to store the augmented data
augmented_train_data = []

# Loop through each row in the original DataFrame
for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing Audio Files"):
    emotion = row['Emotions']
    file_path = row['Path']

    # Load the original audio
    waveform, sample_rate = torchaudio.load(file_path)

    # Apply augmentations
    augmented_waveforms = {
        'add_noise': add_noise(waveform),
        'change_vol': change_vol(waveform),
        'time_masking': time_masking(waveform)
    }

    # Generate and store spectrograms for each augmented audio
    for aug_name, aug_waveform in augmented_waveforms.items():
        spect = spectrogram(aug_waveform, sample_rate).squeeze()
        augmented_train_data.append({
            'Emotions': emotion,
            'Original Path': file_path,
            'Spectrogram': spect,
            'Augmentation': aug_name
        })

# Create a new DataFrame with the augmented data
augmented_df = pd.DataFrame(augmented_train_data)

# Save the DataFrame to a pickle file for future use
augmented_df.to_pickle('/content/drive/My Drive/deep_learn_project/augmented_train_df.pkl')
