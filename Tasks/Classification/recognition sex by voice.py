import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt


torchaudio.set_audio_backend("soundfile")


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


n_fft = 1024
hop_length = 512
n_mels = 64
spectrogram = T.MelSpectrogram(
    n_fft=n_fft,
    hop_length=hop_length,
    center=True,
    power=2.0,
    n_mels=n_mels
)

train_path = r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\Voices\train\train_wav\\"

directory = os.fsencode(train_path)
i = 0
durations = []
mel_spectograms = []
for file in os.listdir(directory):
    i += 1
    if i > 1000:
        break
    filename = os.fsdecode(file)
    waveform, sample_rate = torchaudio.load(train_path + filename)
    durations.append(waveform.shape[1])
    spec = spectrogram(waveform)
    print(spec.shape)
    mel_spectograms.append(spectrogram(waveform))


# plot_waveform(waveform0, sample_rate0)
# plot_specgram(waveform0, sample_rate0)

# define transformation

# Perform transformation
# print("Waveform shape")
# print(waveform0.shape)
# print(waveform1.shape)
# spec0 = spectrogram(waveform0)
# spec1 = spectrogram(waveform1)
# print("spectrogram shape")
# print(spec0.shape)
# print(spec1.shape)
