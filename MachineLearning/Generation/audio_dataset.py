import torch
import numpy as np
import scipy.io.wavfile as wav
from torch.utils.data import Dataset
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, audio_dir, seq_length=100, device='cuda'):
        self.seq_length = seq_length
        self.audio_segments = []
        self.device = device
        
        # Load all wav files from directory
        for wav_file in Path(audio_dir).glob("*.wav"):
            sr, audio = wav.read(str(wav_file))
            audio = audio.astype(float) / np.max(np.abs(audio))
            
            # Split audio into segments
            for i in range(0, len(audio) - seq_length - 1, seq_length):
                self.audio_segments.append(audio[i:i + seq_length + 1])
                
        self.audio_segments = torch.FloatTensor(np.array(self.audio_segments)).to(device)
    
    def __len__(self):
        return len(self.audio_segments)
    
    def __getitem__(self, idx):
        segment = self.audio_segments[idx]
        return segment[:-1], segment[1:]  # input, target