import wave
import numpy as np
from scipy.signal import resample

def downsample_wav(input_file, output_file, target_rate):
    # Open the input WAV file
    with wave.open(input_file, 'rb') as wav_in:
        # Extract original parameters
        params = wav_in.getparams()
        n_channels, sampwidth, framerate, n_frames = params[:4]
        
        # Read the audio data
        audio_data = wav_in.readframes(n_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Reshape the audio data based on the number of channels
        audio_data = np.reshape(audio_data, (n_frames, n_channels))
        
        # Calculate the number of samples in the downsampled audio
        num_samples = int(n_frames * target_rate / framerate)
        
        # Downsample the audio data
        downsampled_data = resample(audio_data, num_samples)
        
        # Convert the downsampled data back to int16
        downsampled_data = downsampled_data.astype(np.int16)
        
        # Flatten the data if it has multiple channels
        downsampled_data = downsampled_data.flatten()
        
        # Write the downsampled audio to the output WAV file
        with wave.open(output_file, 'wb') as wav_out:
            wav_out.setnchannels(n_channels)
            wav_out.setsampwidth(sampwidth)
            wav_out.setframerate(target_rate)
            wav_out.writeframes(downsampled_data.tobytes())

# Example usage
downsample_wav('./taibar2_2_seg_13.wav', 'output_4k.wav', 4000)
