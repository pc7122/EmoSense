import wave
import pyaudio
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram


audio_data = pd.read_csv('./data/audio/data_path.csv')
audio_model = tf.keras.models.load_model(f'./model/audio/audio_cnn_lstm_2.h5')
classes = np.array(['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])


def record_and_save(container, filename, duration=5, sample_rate=22050, channels=2, format_=pyaudio.paInt16):
    p = pyaudio.PyAudio()

    stream = p.open(format=format_,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    with container.status(f"Recording Audio..."):
        st.write(f"Recording {duration - 1} seconds of audio")

        frames = []
        for i in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save to WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(format_))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    st.toast(f"Audio recorded successfully!", icon="😃")


def get_audio_model():
    global audio_model

    with st.sidebar:
        model_name: str = st.selectbox('Models', (
            'Audio CNN LSTM 1',
            'Audio CNN LSTM 1 Extended',
            'Audio CNN LSTM 2',
        ), placeholder='Choose audio model', index=2)

    model_file = '_'.join(model_name.lower().split(' '))
    audio_model = tf.keras.models.load_model(f'./model/audio/{model_file}.h5')


def get_audio_list(dataset):
    paths = audio_data.Path.tolist()
    if dataset == 'RAVDESS':
        return list(filter(lambda x: x.startswith('data'), paths))
    elif dataset == 'SAVEE':
        return list(filter(lambda x: x.startswith('ALL'), paths))
    elif dataset == 'CREMA':
        return list(filter(lambda x: x.startswith('AudioWAV'), paths))

    return list()


def get_actual_emotion(path):
    return audio_data[audio_data.Path == path].Emotions.values[0]


def load_audio_file(filename, duration=3, offset=0, noise=False):
    # load audio file
    y, sr = librosa.load(filename, sr=22050, duration=duration, offset=offset)

    # Trim audio to 3 seconds
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)))

    if noise:
        y = add_noise(y)

    return y, sr


# add noise to the audio signal
def add_noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


# extract mel_spectrogram from the input audio
def get_mel_spectrogram(y, sample_rate):
    mel_spectrogram = melspectrogram(y=y, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=90)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    assert log_mel_spectrogram.shape == (90, 130)
    return log_mel_spectrogram


# save the wave plot of the audio
def save_wave_plot(y, sr):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('./assets/wave_plot.png')


# save the feature plot of the audio
def save_features_plot(y, sr):
    mel_spectrogram = get_mel_spectrogram(y, sr)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./assets/features_plot.png')


def predict_emotion(mel_spectrogram):
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)

    y_pred = audio_model.predict(mel_spectrogram)
    output = np.argmax(y_pred, axis=1)
    confidence = np.max(y_pred)

    return round(confidence*100, 2), classes[output[0]]


def audio_pipeline(audio_path: str, duration=3, offset=0, noise=False, ensemble=False, original_emotion=False):
    y, sr = load_audio_file(audio_path, duration=duration, offset=offset, noise=noise)

    save_wave_plot(y, sr)
    save_features_plot(y, sr)

    mel_spectrogram = get_mel_spectrogram(y, sr)
    confidence, emotion = predict_emotion(mel_spectrogram)

    col1, col2 = st.columns(2)

    with col1:
        st.write('Recorded Audio')
        st.audio(audio_path, format='audio/wav')

        st.write('Mel Spectrogram')
        st.image('./assets/features_plot.png', use_column_width=True)

    with col2:
        st.write('Wave plot')
        st.image('./assets/wave_plot.png', use_column_width=True)
        st.divider()

        subcol1, subcol2 = st.columns(2)

        if original_emotion:
            original_emotion = get_actual_emotion(audio_path.replace('./data/audio/', ''))
            subcol1.metric(label="Original Emotion", value=original_emotion)

        subcol2.metric(label="Predicted Emotion", value=emotion, delta=f'{confidence}%')


