# Speech Emotion Detection API

This repository contains an API for detecting emotions in audio files using machine learning models. The API extracts features such as Mel Spectrogram, MFCCs, Zero Crossing Rate, and RMS Energy from audio files, and predicts emotions using a pre-trained CNN model.

## Features

- **Predict Emotions from Audio**: Detect emotions from audio files provided via a URL.
- **Sliding Window Emotion Prediction**: Apply sliding windows to audio signals for more fine-grained emotion prediction over time.
- **Pre-built Test Audio**: Test the API using pre-built audio samples with known emotions.
- **FastAPI Implementation**: Built with FastAPI for high-performance, asynchronous processing.
- **CORS Support**: Allow cross-origin requests for flexible use cases.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sulaimonmele/a2f.git
   cd a2f
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model and label encoder:
   - Place `best_cnn_model.keras` and `label_encoder_classes.npy` in the root directory.

## Usage

1. Run the API:
   ```bash
   uvicorn main:app --reload
   ```

2. Access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

3. Test the `/predict` endpoint by providing a URL to an audio file. The API will return predicted emotions along with their confidence scores.

### Endpoints

- **`POST /predict`**: Predicts emotions from a provided audio file URL.
  - **Input**: JSON with `url` field containing the audio file URL.
  - **Output**: JSON with predicted emotions and their confidence scores.

- **`POST /predict/v2`**: Predicts emotions using a sliding window approach for more detailed predictions over time.
  - **Input**: JSON with `url` field containing the audio file URL.
  - **Output**: JSON with emotions detected over different timestamps.

- **`GET /test`**: Provides a list of test audio files with known emotions for testing the API.

- **`GET /healthz`**: Health check endpoint to verify the API is running.

- **`GET /`**: Redirects to the API documentation.

- **`GET /git`**: Redirects to the GitHub repository.

## Example Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p1_anger.wav"}'
```

### Example Response

```json
{
  "status": "success",
  "message": "Emotions predicted successfully.",
  "data": {
    "audioDuration": "3.0 seconds",
    "emotions": [
      {
        "timestamp": 0,
        "duration": 3,
        "emotion": "anger",
        "confidence": 0.92
      }
    ]
  }
}
```

## Pre-built Test Audio

The following (Out Of Sample )test audio files can be used for live testing of the API:

- `https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p1_anger.wav` (Emotion: Anger)
- `https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p1_neutral.wav` (Emotion: Neutral)
- `https://github.com/sulaimonmele/a2f/raw/main/TestFiles/english_voice_male_p2_amazement.wav` (Emotion: Amazement)

The following (In Sample )test audio files can be used for live testing of the API:

- `https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1034_ITS_HAP_XX.wav` (Emotion: HAP)
- `https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1048_IWL_ANG_XX.wav` (Emotion: ANG)
- `https://github.com/sulaimonmele/a2f/raw/main/CremaDTestFiles/1049_TAI_NEU_XX.wav` (Emotion: NEU)

More test files are available at the `/test` endpoint.

## License

This project is licensed under the MIT License.

---

Feel free to contribute by submitting pull requests or issues on the [GitHub repository](https://github.com/sulaimonmele/a2f).
