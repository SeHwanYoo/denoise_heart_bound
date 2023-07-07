import numpy as np
import librosa
import tensorflow as tf
import scipy.signal

import soundfile as sf


# Load the audio file
def load_audio(file_path, sample_rate=22050):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

# Preprocess the audio
def preprocess_audio(audio, sample_rate):
    # Convert audio to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=1024, win_length=2048, window='hann')
    
    # Apply log-scale to the mel spectrogram
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize the mel spectrogram
    mel_spec_db_norm = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
    
    return mel_spec_db_norm

# Apply lowpass filter to the mel spectrogram
def apply_lowpass_filter(mel_spec, cutoff_freq, sample_rate):
    nyquist_freq = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = scipy.signal.butter(5, normalized_cutoff, btype='lowpass')
    filtered_mel_spec = scipy.signal.lfilter(b, a, mel_spec)
    return filtered_mel_spec

# Denoise the mel spectrogram using a pre-trained model
def denoise_mel_spectrogram(mel_spec):
    # Load the pre-trained denoising model
    model = tf.keras.models.load_model('denoising_model.h5')
    
    # Expand dimensions for model input
    mel_spec_input = np.expand_dims(mel_spec, axis=0)
    
    # Predict denoised mel spectrogram
    denoised_mel_spec = model.predict(mel_spec_input)
    
    return denoised_mel_spec[0]

# Convert mel spectrogram back to audio
def mel_to_audio(mel_spec, sample_rate):
    # Inverse of log-scale on mel spectrogram
    mel_spec = librosa.db_to_power(mel_spec)
    
    # Convert mel spectrogram to audio
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sample_rate)
    
    return audio

# Divide audio into 1-minute segments
def divide_audio(audio, sample_rate, segment_length=60):
    segment_samples = segment_length * sample_rate
    num_segments = len(audio) // segment_samples
    segments = np.array_split(audio[:num_segments * segment_samples], num_segments)
    return segments

# Build denoising autoencoder model
def build_denoising_autoencoder(input_shape):
    # model = tf.keras.models.Sequential()
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # Decoder
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    return model


input_audio, sr = librosa.load('/Users/sehwan/Desktop/datasets/bowel_sound/output.wav')

# Divide audio into 1-minute segments
segments = divide_audio(input_audio, sr, segment_length=60)

denoised_segments = []
for segment in segments:
    # Preprocess each segment
    segment = apply_lowpass_filter(segment, cutoff_freq=2000, sample_rate=sr)
    
    mel_spec = preprocess_audio(segment, sr)
    
    # Apply lowpass filter to the mel spectrogram
    # filtered_mel_spec = apply_lowpass_filter(mel_spec, cutoff_freq=2000, sample_rate=sr)
    
    # print(filtered_mel_spec.shape)
    # print(mel_spec.shape)
    # exit()
    
    denoised_segments.append(mel_spec)
    
    # # Denoise each segment
    # denoised_mel_spec = denoise_mel_spectrogram(filtered_mel_spec)
    # print(denoised_mel_spec.shape)
    # exit()
    # # Convert denoised mel spectrogram back to audio
    # denoised_segment = mel_to_audio(denoised_mel_spec, sr)
    # denoised_segments.append(denoised_mel_spec)
    # 
denoised_segments = np.array(denoised_segments).reshape([-1, 128, 2584])

model = build_denoising_autoencoder(denoised_segments.shape[1:])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the denoising autoencoder
model.fit(denoised_segments, denoised_segments, epochs=5, batch_size=1)

preds = model.predict(denoised_segments, batch_size=32)

# for idx, pred in enumerate(preds):
#     reduced_noise = mel_to_audio(pred, sr)
    
    # librosa.output.write_wav(f'denoised_audio_{str(idx)}.wav', audio, sr)
sf.write(f'denoised_audio.wav', mel_to_audio(preds[0], sr), sr)

    

# print(denoised_segments.shape) 
# Concatenate denoised segments into a single audio
# denoised_audio = np.concatenate(denoised_segments)
# denoised_segments = np.array(denoised_segments)
# print('-' * 50)
# print(denoised_segments.shape)