import torch as t
import torch
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import subprocess
import os
import glob
import ffmpeg
import io


def convert_mp3_to_tensor(input_file):
    try:
        # Run FFmpeg command and capture output in memory
        process = subprocess.Popen(
            [
                'ffmpeg',
                '-i', input_file,
                '-f', 'wav',
                'pipe:'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = process.communicate()

        # Check if FFmpeg command was successful
        if process.returncode != 0:
            print(f"Error: FFmpeg command failed with return code {process.returncode}")
            return None

        # Convert output to BytesIO object
        wav_data = io.BytesIO(out)

        # Load waveform directly from BytesIO object
        waveform, sample_rate = torchaudio.load(wav_data)

        return waveform, sample_rate
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None


def convert_all_mp3_to_tensor(mp3_directory):
    tensor_list = []
    
    # Use glob to find all MP3 files in the directory
    mp3_files = glob.glob(os.path.join(mp3_directory, '*.mp3'))
    
    for mp3_file in mp3_files:
        # Convert MP3 to Torch tensor
        tensor = convert_mp3_to_tensor(mp3_file)
        if tensor is not None:
            tensor_list.append(tensor)
    
    # Concatenate tensors along time dimension (assuming they have the same shape)
    if tensor_list:
        return torch.cat(tensor_list, dim=1)
    else:
        return None
    
def Audio2Tensor(file_path, sr, n_mels, hop_length):
    # Check if the file is an MP3
    if file_path.endswith('.mp3'):
        # Convert MP3 to Torch tensor
        waveform, sample_rate = convert_mp3_to_tensor(file_path)
    else:
        # Load waveform from file
        waveform, sample_rate = torchaudio.load(file_path)
    
    # Resample if necessary
    if sample_rate != sr:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
    
    # Convert to Mel Spectrogram
    melspec_transform = transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        hop_length=hop_length
    )
    mel_spectrogram = melspec_transform(waveform)
    
    # Convert to log scale (dB)
    log_melspec = transforms.AmplitudeToDB()(mel_spectrogram)
    
    # Return the resulting tensor
    return log_melspec

class AudioDataset(Dataset):

    def __init__(self, directory, sr=22050, n_mels=64, hop_length=512):
        self.directory = directory
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.file_paths = glob.glob(os.path.join(directory, '*.mp3'))

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            log_melspec = Audio2Tensor(file_path, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
            return log_melspec, file_path
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None, file_path
        
    # Collate function to handle None values

def collate_fn(batch):

    # Filter out None values
    batch = [(data, path) for data, path in batch if data is not None]
    if not batch:
        return torch.tensor([]), []
    
    # Separate data and file paths
    data, paths = zip(*batch)
    return torch.stack(data), paths


if __name__ == "__main__":

    mp3_directory = 'Mp3'
    dataset = AudioDataset(mp3_directory)

    # Create a DataLoader with the custom collate function
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Iterate through the dataset
    for i, (data, file_paths) in enumerate(data_loader):
        print(f"Batch {i+1}")
        if data.numel() > 0:
            print(data.shape)
        else:
            print(f"Error in batch {i+1}: {file_paths}")
























