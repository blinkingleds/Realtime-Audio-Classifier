import os 
import torch
import shutil
import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import torchaudio

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, sample_rate=16000, duration_sec=3.0):
        self.data_path = data_path
        self.files = []
        self.labels = []
        self.classes = ['music','speech','noise']
        self.sr = sample_rate
        self.target_length = int(sample_rate * duration_sec)
        
        # gather all file paths and corresponding labels
        for idx,label in enumerate(self.classes):
            class_path = os.path.join(data_path, label)
            for dirpath, _, filenames in os.walk(class_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    self.files.append(file_path)
                    self.labels.append(idx)
    

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Load audio file
        signal, _ = librosa.load(file_path, sr=self.sr, mono=True)

        # Handle empty or corrupted audio files
        if signal.size == 0:
            print(f"Warning: Loaded empty signal for file: {file_path}. Replacing with zeros.")
            signal = np.zeros((self.target_length,), dtype=np.float32)

        # Normalize signal
        max_amp = np.max(np.abs(signal))
        if max_amp > 0.005: 
            signal = signal / max_amp
        
        
        n_samples = len(signal)

        if n_samples > self.target_length:
            signal = signal[0:self.target_length]
        elif n_samples < self.target_length:
            padding = self.target_length - n_samples
            signal = np.pad(signal, (0, padding), 'constant')
            
        
        # convert to torch tensor
        signal_torch = torch.from_numpy(signal)
            
            
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64,
        )(signal_torch)

        log_mel_spec = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)(mel_spec)
        

        tensor_mel_spec = log_mel_spec.unsqueeze(0) 
       
        
        return tensor_mel_spec, label


