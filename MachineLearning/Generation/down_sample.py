import wave
import numpy as np
from scipy.signal import resample
import os
import shutil

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.signal import resample as gpu_resample
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU acceleration not available. Using CPU instead.")

def downsample_wav(input_file, output_dir, target_rate=4000):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, base_name)
    
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
        
        # If GPU is available, use it for resampling
        if GPU_AVAILABLE:
            try:
                # Transfer data to GPU
                gpu_audio = cp.asarray(audio_data)
                
                # Calculate the number of samples in the downsampled audio
                num_samples = int(n_frames * target_rate / framerate)
                
                # Downsample the audio data on GPU
                downsampled_data = gpu_resample(gpu_audio, num_samples)
                
                # Transfer back to CPU and convert to int16
                downsampled_data = cp.asnumpy(downsampled_data).astype(np.int16)
            except Exception as e:
                print(f"GPU processing failed, falling back to CPU: {str(e)}")
                # Fallback to CPU processing
                num_samples = int(n_frames * target_rate / framerate)
                downsampled_data = resample(audio_data, num_samples)
                downsampled_data = downsampled_data.astype(np.int16)
        else:
            # CPU processing
            num_samples = int(n_frames * target_rate / framerate)
            downsampled_data = resample(audio_data, num_samples)
            downsampled_data = downsampled_data.astype(np.int16)
        
        # Flatten the data if it has multiple channels
        downsampled_data = downsampled_data.flatten()
        
        # Write the downsampled audio to the output WAV file
        with wave.open(output_file, 'wb') as wav_out:
            wav_out.setnchannels(n_channels)
            wav_out.setsampwidth(sampwidth)
            wav_out.setframerate(target_rate)
            wav_out.writeframes(downsampled_data.tobytes())

def process_directory(input_dir, output_dir, target_rate=4000):
    """Process all WAV files in the input directory."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = 0
    errors = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            input_path = os.path.join(input_dir, filename)
            try:
                downsample_wav(input_path, output_dir, target_rate)
                processed_files += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                errors += 1
    
    print(f"Processing complete. Successfully processed {processed_files} files. Errors: {errors}")

if __name__ == "__main__":
    input_dir = "F:\\Sean\\VerilogWithML\\raw"
    output_dir = "F:\\Sean\\VerilogWithML\\downsample"
    
    try:
        process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            target_rate=4000
        )
    except Exception as e:
        print(f"Error: {str(e)}")
