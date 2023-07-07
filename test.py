import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle
# from scipy.signal import stft, periodogram
from scipy import signal, fft
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, welch, lfilter
import scipy.signal

import tensorflow as tf
import librosa.display
import soundfile
import wave
import IPython.display as ipd

from scipy.io.wavfile import write


input_audio, sr = librosa.load('/Users/sehwan/Desktop/datasets/bowel_sound/output.wav')
input_audio = input_audio[:sr*10]

S_fft = np.fft.fft(input_audio)
ampl = abs(S_fft) * (2/len(S_fft))
freq = np.fft.fftfreq(len(S_fft), 1/sr)


plt.xlim(0, 50)
plt.stem(freq, ampl)
plt.grid(True)
plt.savefig('./1.png') 