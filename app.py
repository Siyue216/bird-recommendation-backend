from flask import Flask, request, make_response
import torch
import torchaudio
import numpy as np
import os
from flask_cors import CORS
import librosa
import soundfile as sf
import torch.nn as nn
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the label mapping
train_df = pd.read_csv('train.csv')
unique_labels = sorted(train_df['primary_label'].unique())
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fourth conv block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate the size of the flattened features
        self._to_linear = None
        self._get_conv_output_size()
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _get_conv_output_size(self):
        # Create a dummy input to calculate the size
        x = torch.randn(1, 1, 64, 500)  # Updated input size (64 mel bands, 500 time steps)
        x = self.conv1(x)  # 32 x 32 x 250
        x = self.conv2(x)  # 64 x 16 x 125
        x = self.conv3(x)  # 128 x 8 x 62
        x = self.conv4(x)  # 256 x 4 x 31
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize model and load weights
model = CNN(num_classes=len(unique_labels))  # Use actual number of classes
model.load_state_dict(torch.load('model/best_model.pth', map_location=torch.device('cpu')))
model.eval()

def process_audio(audio_path):
    # Load and process audio with reduced sample rate
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Create mel spectrogram with fewer mel bands
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Pad or truncate to max_length (500)
    max_length = 500
    if mel_spec_db.shape[1] < max_length:
        pad_width = max_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_length]
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
    
    # Convert to tensor and add batch and channel dimensions
    mel_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)  # Add both batch and channel dimensions
    
    # Apply normalization
    mel_tensor = (mel_tensor - 0.485) / 0.229
    
    return mel_tensor

@app.route('/', methods=["GET"])
def home():
    return make_response("Backend for Bird Voice Classification System", 200)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the first file from the request, regardless of the key
        if not request.files:
            return make_response({
                "Error": {
                    "code": 400,
                    "message": "No audio file provided"
                }
            }, 400)
        
        # Get the first file from the ImmutableMultiDict
        audio_file = next(iter(request.files.values()))
        
        if audio_file.filename == '':
            return make_response({
                "Error": {
                    "code": 400,
                    "message": "No selected file"
                }
            }, 400)
        
        if not audio_file.filename.endswith('.ogg'):
            return make_response({
                "Error": {
                    "code": 400,
                    "message": "Only .ogg files are supported"
                }
            }, 400)
        
        # Save the uploaded file temporarily
        temp_path = "temp_audio.ogg"
        audio_file.save(temp_path)
        
        # Process the audio
        audio_tensor = process_audio(temp_path)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(audio_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()
            bird_name = idx_to_label[predicted_class]
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return make_response({
            "Success": {
                "code": 200,
                "prediction": bird_name
            }
        }, 200)
        
    except Exception as e:
        return make_response({
            "Error": {
                "code": 500,
                "message": str(e)
            }
        }, 500)

if __name__ == '__main__':
    app.run(debug=True)