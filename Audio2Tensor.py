import torch as t
import torchaudio
import torchaudio.transforms as transforms
from pydub import AudioSegment

def Audio2Tensor(file_path,wav_path,sr=22050,n_mels=128,hop_length =512):

    #load audio file
    audio = AudioSegment.from_mp3(file_path, parameters=["-analyzeduration", "100M", "-probesize", "100M"])
    # Export the audio as a WAV file
    audio.export(wav_path, format="wav")
    waveform,sample_Rate = torchaudio.load(wav_path)
    # -> check for resampling
    if sample_Rate != sr:
        resampler = transforms.Resample(orig_freq=sample_Rate,new_freq=sr)
        waveform = resampler(waveform)
        sample_Rate = sr

    #convert to mel spectrogram
    melspec_transform = transforms.MelSpectrogram(
        sample_rate=sample_Rate,
        n_mels = n_mels,
        hop_length=hop_length
    )
    mel_spectrogram= melspec_transform(waveform)

    #convert to log scale or DB
    log_melspec = transforms.AmplitudeToDB()(mel_spectrogram)

    #return the resulting data
    return log_melspec

if __name__ == "__main__":
    file_path = 'Self_Destruction.mp3'
    wav_path = 'Self_Destruction.wav'
    tensor = Audio2Tensor(file_path,wav_path)
    print(tensor.shape) 














































