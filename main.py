import asyncio
import tempfile
import os
import requests
import librosa
import numpy as np
import resampy
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for all origins (you can specify a list of domains in place of "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this if you want to restrict)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Global configuration
WINDOW_DURATION = 3  # 5Window size in seconds
HOP_DURATION = 1     # 2Hop size in seconds (overlapping windows)
CHUNK_DURATION = 3  # 3Duration of the audio chunk in seconds

# Load the trained model
model = load_model('best_cnn_model.keras')  # Replace with your model file

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)


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
    chunk_size = int(CHUNK_DURATION * sr)  # Convert to integer to ensure it can be used in slicing operations
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



# Function to predict emotions in audio windows asynchronously
async def async_predict_emotion(window, sr):
    features = extract_features(window, sr)
    features = features.reshape(1, features.shape[0], features.shape[1], 1)
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    emotion = label_encoder.classes_[emotion_index]
    confidence = prediction[0][emotion_index]
    return emotion, round(float(confidence), 2)


# Function to apply sliding window to the audio signal
def apply_sliding_window(signal, sr, window_size=WINDOW_DURATION, hop_size=HOP_DURATION):
    window_size_samples = int(window_size * sr)
    hop_size_samples = int(hop_size * sr)
    windows = []
    for start in range(0, len(signal) - window_size_samples + 1, hop_size_samples):
        window = signal[start:start + window_size_samples]
        windows.append((window, start / sr, (start + window_size_samples) / sr))  # Tuple of (window, start_time, end_time)
    return windows


# Function to predict emotions using sliding windows
async def predict_emotions_with_sliding_window(signal, sr):
    windows = apply_sliding_window(signal, sr)
    tasks = [async_predict_emotion(window, sr) for window, _, _ in windows]
    predictions = await asyncio.gather(*tasks)

    # Create a list of dictionaries with timestamp, emotion, and confidence
    results = []
    for i, ((_, start_time, end_time), (emotion, confidence)) in enumerate(zip(windows, predictions)):
        results.append({
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "emotion": emotion,
            "confidence": confidence
        })
    return results



# Function to download the audio file
def download_audio_file(url: str) -> str:
    """Download an audio file from the given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    else:
        raise HTTPException(status_code=400, detail=f"Failed to download the file. Status code: {response.status_code}")

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    """API endpoint to predict emotions from an audio file provided via URL."""
    audio_file_path = None  # Initialize here to ensure it exists
    try:
        audio_file_path = download_audio_file(data.url)
        with open(audio_file_path, 'rb') as audio_file:
            response = predict_emotions(audio_file)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during emotion prediction: {str(e)}")
    finally:
        if audio_file_path:
            os.remove(audio_file_path)

# POST request to predict emotions using sliding windows
@app.post("/predict/v2")
async def predict_v2(data: URLInput):
    audio_file_path = None  # Initialize the variable to store the downloaded audio file
    try:
        audio_file_path = download_audio_file(data.url)
        # Load the audio file
        signal, sr = librosa.load(audio_file_path, sr=None)
        audio_duration = librosa.get_duration(y=signal, sr=sr)

        # Run predictions using sliding window
        emotions = await predict_emotions_with_sliding_window(signal, sr)

        # Create response JSON
        response = {
            "status": "success",
            "message": "Emotions predicted successfully.",
            "data": {
                "audioDuration": str(round(audio_duration, 2)) + " seconds",
                "emotions": emotions
            }
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during emotion prediction: {str(e)}")
    finally:
        if audio_file_path:
            os.remove(audio_file_path)  # Clean up the temporary file



@app.get("/predict")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/predict/v2")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/healthz")
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.get("/git")
async def redirect_to_github():
    return RedirectResponse(url="https://github.com/sulaimonmele/a2f")

@app.get("/unitydemo")
async def redirect_to_github():
    return RedirectResponse(url="https://play.unity.com/en/user/3221ee49-c016-4f4d-8b5c-da38b7c911a7")


@app.get("/")
async def root():
    #return {"message": "Hello, Welcome To Speech Emotion Detection API!"}
    return RedirectResponse(url="/docs")

@app.get("/test")
def get_test_audio_files():
    test_audio_files = [
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p1_anger.wav",
            "emotion": "anger"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p1_neutral.wav",
            "emotion": "neutral"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p2_amazement.wav",
            "emotion": "amazement"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p2_disgust.wav",
            "emotion": "disgust"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p2_neutral.wav",
            "emotion": "neutral"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p2_sadness.wav",
            "emotion": "sadness"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p3_anger.wav",
            "emotion": "anger"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p3_fear.wav",
            "emotion": "fear"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p3_grief.wav",
            "emotion": "grief"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p3_joy.wav",
            "emotion": "joy"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p3_neutral.wav",
            "emotion": "neutral"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p3_outofbreath.wav",
            "emotion": "out of breath"
        },
        {
            "url": "https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p3_pain.wav",
            "emotion": "pain"
        }
    ]

    CremaD_audio_files = [
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1007_ITS_SAD_XX.wav",
        "emotion": "SAD"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1009_IEO_ANG_MD.wav",
        "emotion": "ANG"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1012_ITS_SAD_XX.wav",
        "emotion": "SAD"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1021_ITS_HAP_XX.wav",
        "emotion": "HAP"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1034_ITS_HAP_XX.wav",
        "emotion": "HAP"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1048_IWL_ANG_XX.wav",
        "emotion": "ANG"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1049_TAI_NEU_XX.wav",
        "emotion": "NEU"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1073_IEO_DIS_MD.wav",
        "emotion": "DIS"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1078_TSI_NEU_XX.wav",
        "emotion": "NEU"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1080_IOM_NEU_XX.wav",
        "emotion": "NEU"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1082_MTI_FEA_XX.wav",
        "emotion": "FEA"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1085_MTI_SAD_XX.wav",
        "emotion": "SAD"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1086_ITS_DIS_XX.wav",
        "emotion": "DIS"
    },
    {
        "url": "https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1090_MTI_SAD_XX.wav",
        "emotion": "SAD"
    }
    ]

    return {"Real Live (Out Of Training) Test Audio Samples": test_audio_files,"CREMAD(Data Set) Test Audio Samples": CremaD_audio_files}
