import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import pandas as pd
from tqdm import tqdm

def spectrogram(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=128)(waveform.squeeze())
    mel_spectrogram_db = AmplitudeToDB()(mel_spectrogram)
    return mel_spectrogram_db

tqdm.pandas()  # Initializes tqdm for pandas
Crema_df['Spectrogram'] = Crema_df['Path'].progress_apply(spectrogram)
Crema_df.to_pickle('/content/drive/My Drive/deep_learn_project/Crema_spect_df.pkl')
