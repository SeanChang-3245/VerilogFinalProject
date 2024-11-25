import os
import torch
import librosa
import shutil
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pretrained model and move to GPU
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to(device)
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Define the directory containing audio files
audio_dir = "F:\\Sean\\classified_data\\taibar2"

# Define the output directory for high confidence files
output_dir = "F:\\Sean\\VerilogWithML"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to check if the confidence score for classes 111, 112, or 113 is higher than 0.15
def check_specific_classes_confidence(audio_path):
    # Load the audio file
    audio, sampling_rate = librosa.load(audio_path, sr=None)

    # Extract features
    inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    # Move inputs to GPU
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Move logits to CPU for further processing
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu()

        # Check confidence scores for classes 111, 112, and 113
        class_ids = [111, 112, 113]
        for class_id in class_ids:
            confidence = probabilities[0][class_id].item()
            if confidence > 0.20:
                return True, class_id, confidence

    return False, None, None

# Iterate over all audio files in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):  # Assuming audio files are in .wav format
        audio_path = os.path.join(audio_dir, filename)
        is_confident, class_id, confidence = check_specific_classes_confidence(audio_path)
        if is_confident:
            print(f"{filename}: Class {class_id} confidence is higher than 0.15 (Confidence: {confidence})")
            # Copy file to output directory
            output_path = os.path.join(output_dir, f"class{class_id}_{filename}")
            shutil.copy2(audio_path, output_path)
            print(f"Copied to: {output_path}")
        else:
            print(f"{filename}: No class among 111, 112, 113 has confidence higher than 0.15")