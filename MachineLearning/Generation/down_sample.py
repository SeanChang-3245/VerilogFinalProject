import librosa
import soundfile as sf
import numpy as np

def downsample_wav(input_file, output_file, original_rate=16000, target_rate=4000):
    # Read the audio file
    audio_data, sample_rate = librosa.load(input_file, sr=original_rate, mono=True)
    
    # Perform resampling
    # downsampled = librosa.resample(audio_data, orig_sr=original_rate, target_sr=target_rate)
    
    
    # Save the downsampled audio
    sf.write(output_file, audio_data, target_rate, subtype='PCM_16')

# Example usage
if __name__ == "__main__":
    input_file = "F:\\Sean\\classified_data\\taibar2\\taibar2_1_seg_22.wav"
    output_file = "output_4khz.wav"
    downsample_wav(input_file, output_file)
