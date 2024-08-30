import tempfile
import os
import requests
from flask import Flask, request, jsonify
import librosa
import numpy as np
import resampy
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

app = Flask(__name__)

# Global configuration
CHUNK_DURATION = 4  # Duration of the audio chunk in seconds

# Load the trained model
model = load_model('best_cnn_model.keras')  # Replace with your model file

# Load the label encoder
label_encoder = LabelEncoder()
# Load the label encoder classes
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)  # Replace with your label encoder classes file

# Function to extract features from audio
def extract_features(signal, sr):
    # Resample to 22050 Hz
    signal_resampled = resampy.resample(signal, sr, 22050)

    # Extract Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal_resampled, sr=22050, n_mels=128)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=signal_resampled, sr=22050, n_mfcc=13)

    # Extract Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal_resampled)

    # Extract RMS Energy
    rms = librosa.feature.rms(y=signal_resampled)

    # Pad or truncate MFCCs to a fixed length
    max_length = 100
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        zcr = np.pad(zcr, pad_width=((0, 0), (0, pad_width)), mode='constant')
        rms = np.pad(rms, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
        mel_spectrogram = mel_spectrogram[:, :max_length]
        zcr = zcr[:, :max_length]
        rms = rms[:, :max_length]

    # Stack MFCCs, Mel Spectrogram, ZCR, and RMS into a single feature vector
    features = np.vstack((mfccs, mel_spectrogram, zcr, rms))
    return features

# Function to predict emotions in audio chunks
def predict_emotions(audio_file):
    # Load the audio file
    signal, sr = librosa.load(audio_file, sr=None)

    # Calculate audio duration
    audio_duration = librosa.get_duration(y=signal, sr=sr)

    # Create chunks of CHUNK_DURATION seconds
    chunk_size = CHUNK_DURATION * sr  # CHUNK_DURATION seconds in samples
    chunks = []
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i + chunk_size]
        if len(chunk) < chunk_size:
            # Pad the last chunk if it's shorter than chunk_size
            pad_width = chunk_size - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), mode='constant')
        chunks.append(chunk)

    # Predict emotions for each chunk
    emotions = []
    for i, chunk in enumerate(chunks):
        features = extract_features(chunk, sr)
        features = features.reshape(1, features.shape[0], features.shape[1], 1)
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction)
        emotion = label_encoder.classes_[emotion_index]
        confidence = prediction[0][emotion_index]
        emotions.append({
            "timestamp": i * CHUNK_DURATION,  # Timestamp in seconds
            "duration": CHUNK_DURATION,  # Duration of each chunk (CHUNK_DURATION seconds)
            "emotion": emotion,
            "confidence": round(float(confidence), 2)
        })

    # Create the response JSON
    response = {
        "status": "success",
        "message": "Emotions predicted successfully.",
        "data": {
            "audioDuration": str(round(audio_duration, 2)) + " seconds",
            "emotions": emotions
        }
    }
    return response

# Function to download the audio file from a URL
def download_audio_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    else:
        raise Exception(f"Failed to download the file. Status code: {response.status_code}")

# Define the API route
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files and 'url' not in request.json:
        return jsonify({"error": "No audio file or URL provided."}), 400

    audio_file = None
    if 'audio' in request.files:
        audio_file = request.files['audio']
    elif 'url' in request.json:
        try:
            url = request.json['url']
            audio_file_path = download_audio_file(url)
            audio_file = open(audio_file_path, 'rb')
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if audio_file is None or (hasattr(audio_file, 'filename') and audio_file.filename == ''):
        return jsonify({"error": "No selected file."}), 400

    try:
        response = predict_emotions(audio_file)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'url' in request.json:
            os.remove(audio_file.name)  # Clean up the temporary file if it was downloaded

#Define the home route
@app.route("/")
def hello_world():
    return "<p>Hello, Welcome To Speech Emotion Detection API!</p>"
# Run the Flask app
if __name__ == '__main__':
    app.run()
