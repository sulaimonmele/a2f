
# Speech Emotion Detection API

This project provides a RESTful API for predicting emotions from audio files. The API uses a pre-trained Convolutional Neural Network (CNN) model to analyze chunks of audio and return the predicted emotion for each chunk. The model processes audio in chunks of 3 seconds and provides the emotion with the highest confidence for each segment.

## Features

- **Audio Emotion Prediction**: Upload an audio file via URL, and the API returns a prediction of emotions for each 3-second chunk of the audio.
- **Health Check Endpoint**: Easily check if the API is running.
- **Documentation**: Automatically generated API documentation.

## Requirements

- Python 3.7 or higher
- FastAPI
- Keras
- TensorFlow
- librosa
- resampy
- scikit-learn
- numpy
- requests
- pydantic

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/speech-emotion-detection-api.git
   cd speech-emotion-detection-api
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model and label encoder:**
   - Place your trained CNN model in the root directory of the project and name it `best_cnn_model.keras`.
   - Place the `label_encoder_classes.npy` file in the root directory as well.

## Running the API

1. **Start the API:**
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the API documentation:**
   - Navigate to `http://127.0.0.1:8000/docs` in your web browser to view the automatically generated Swagger documentation.

## API Endpoints

- **POST /predict**
  - Predict emotions from an audio file provided via URL.
  - **Request Body:**
    ```json
    {
      "url": "https://example.com/audiofile.wav"
    }
    ```
  - **Response:**
    ```json
    {
      "status": "success",
      "message": "Emotions predicted successfully.",
      "data": {
        "audioDuration": "12.34 seconds",
        "emotions": [
          {
            "timestamp": 0,
            "duration": 3,
            "emotion": "happy",
            "confidence": 0.85
          },
          {
            "timestamp": 3,
            "duration": 3,
            "emotion": "sad",
            "confidence": 0.92
          },
          ...
        ]
      }
    }
    ```

- **GET /healthz**
  - Check the health status of the API.
  - **Response:**
    ```json
    {
      "status": "ok"
    }
    ```

- **GET /**
  - Redirect to the API documentation.

## Project Structure

- `main.py`: Contains the API logic.
- `best_cnn_model.keras`: The pre-trained CNN model.
- `label_encoder_classes.npy`: Numpy file containing the label encoder classes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
