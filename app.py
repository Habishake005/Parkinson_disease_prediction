import streamlit as st
import librosa
import numpy as np
import gzip
import pickle
import tensorflow as tf
import joblib
import os
import tensorflow as tf
from tensorflow.keras import layers


# Constants
T = 20

# Custom Attention layer
class Attention(layers.Layer):
    def call(self, inputs):
        score = tf.nn.tanh(inputs)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

class TCNBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout=0.2, **kwargs):
        super(TCNBlock, self).__init__(**kwargs)  # Pass kwargs to base Layer
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal',
                                   dilation_rate=dilation_rate, activation='relu')
        self.dropout1 = layers.Dropout(dropout)
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal',
                                   dilation_rate=dilation_rate, activation='relu')
        self.dropout2 = layers.Dropout(dropout)
        self.downsample = layers.Conv1D(filters, 1)
        self.activation = layers.Activation('relu')

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        res = self.downsample(inputs)
        return self.activation(x + res)


# Load scaler and models
with gzip.open('scaler (1).gz', 'rb') as f:
    scaler = pickle.load(f)

lstm_model = tf.keras.models.load_model('final_lstm_model.keras', custom_objects={'Attention': Attention})
tcn_model = tf.keras.models.load_model('final_tcn_model.keras', custom_objects={'TCNBlock': TCNBlock})
meta_model = joblib.load('meta_model.joblib')

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Zero Crossing Rate (mean)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Root Mean Square Energy (mean)
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Spectral Centroid (mean)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Spectral Bandwidth (mean)
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    # Spectral Roll-off (mean)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Harmonic-to-Noise Ratio (approx)
    y_harmonic = librosa.effects.harmonic(y)
    y_percussive = librosa.effects.percussive(y)
    energy_harmonic = np.sum(y_harmonic ** 2)
    energy_percussive = np.sum(y_percussive ** 2)
    hnr = energy_harmonic / (energy_percussive + 1e-6)
    
    # MFCC mean of first 3 coefficients (to keep feature count low)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
    mfccs_mean = np.mean(mfccs, axis=1)  # 3 features
    
    # rpde approximation, keep as 0.0 for consistency
    rpde = 0.0
    
    features = np.hstack([
        zcr, rms, spec_centroid, spec_bw, rolloff,
        hnr,
        mfccs_mean,
        rpde
    ])
    
    return features


# Prediction function
def predict_from_audio(file_path):
    features = extract_features(file_path).reshape(1, -1)
    X_seq = np.tile(features, (T, 1)).reshape(1, T, -1)
    X_scaled = scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

    lstm_preds = lstm_model.predict(X_scaled).flatten()
    tcn_preds = tcn_model.predict(X_scaled).flatten()

    X_meta = np.column_stack([lstm_preds, tcn_preds])
    meta_preds = meta_model.predict(X_meta)

    return np.mean(meta_preds)

def classify_severity(prediction):
    if prediction < 60:
        return "Mild","😐"
    elif prediction < 140:
        return "High", "😐"
    else:
        return "Severe", "😕"
    

# Streamlit UI
st.set_page_config(page_title="Parkinson's UPDRS Predictor", layout="centered")
st.title("🧠 Parkinson's Disease Prediction")
st.markdown("""
This application uses advanced AI models to predict the **motor_UPDRS** (Unified Parkinson's Disease Rating Scale) score
from your **voice recording**. By analyzing the acoustic features of speech, we can estimate the severity of motor symptoms in Parkinson’s Disease.

**Usage Instructions:**
1. Record a short **.wav** audio file of your voice.
2. Upload it using the uploader below.
3. Get a predicted motor_UPDRS score instantly!

This project demonstrates the potential of AI and audio analytics in **remote healthcare and monitoring** for Parkinson's Disease patients.
""")

uploaded_file = st.file_uploader('Upload your .wav file', type=["wav"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner('🔍 Analyzing voice features...'):
        try:
            prediction = predict_from_audio("temp_audio.wav")
            severity,emoji = classify_severity(prediction)
            st.subheader("Results")
            st.metric("Predicted motor_UPDRS:", f"{prediction:.2f}")
            st.metric("Severity",f"{severity} {emoji}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    os.remove("temp_audio.wav")
