import os
import torch
import librosa
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# Load the pretrained model and feature extractor
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Define the directory containing audio files
audio_dir = "C:\\ML_Data\\VerilogWithML\\color"

# Function to check if the confidence score for classes 111, 112, or 113 is higher than 0.15
def check_specific_classes_confidence(audio_path):
    # Load the audio file
    audio, sampling_rate = librosa.load(audio_path, sr=None)

    # Extract features
    inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Check confidence scores for classes 111, 112, and 113
        class_ids = [111, 112, 113]
        for class_id in class_ids:
            confidence = probabilities[0][class_id].item()
            if confidence > 0.15:
                return True, class_id, confidence

    return False, None, None

# Iterate over all audio files in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):  # Assuming audio files are in .wav format
        audio_path = os.path.join(audio_dir, filename)
        is_confident, class_id, confidence = check_specific_classes_confidence(audio_path)
        if is_confident:
            print(f"{filename}: Class {class_id} confidence is higher than 0.15 (Confidence: {confidence})")
        else:
            print(f"{filename}: No class among 111, 112, 113 has confidence higher than 0.15")