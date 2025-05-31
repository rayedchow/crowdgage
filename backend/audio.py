# Fix OpenMP error first
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Explicitly import numpy first
import numpy as np

# Other imports
import sounddevice as sd
import time
import sys
import math

# Now import whisper after numpy and torch are loaded
import whisper
from text import score_presentation_engagement

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds
audio_buffer = []

# Function to list available audio input devices
def list_microphones():
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # It's an input device
            print(f"[{i}] {device['name']}")
            input_devices.append(i)
    
    return input_devices

# Function to select microphone
def select_microphone():
    input_devices = list_microphones()
    
    if not input_devices:
        print("No input devices found!")
        sys.exit(1)
    
    # Check for HyperX microphone
    devices = sd.query_devices()
    hyperx_device_id = None
    
    for i in input_devices:
        device = devices[i]
        # Case-insensitive check for 'hyperx' in the device name
        if 'hyperx' in device['name'].lower():
            hyperx_device_id = i
            print(f"Found and automatically selected HyperX microphone: {device['name']}")
            return hyperx_device_id
    
    # Use default device if only one is available
    if len(input_devices) == 1:
        print(f"Using the only available input device: {devices[input_devices[0]]['name']}")
        return input_devices[0]
    
    # Ask user to select a device if HyperX not found
    print("HyperX microphone not found. Please select from available devices:")
    while True:
        selection = input("\nSelect microphone by number (or press Enter for default): ")
        
        if selection == "":
            default_device = sd.default.device[0]
            print(f"Using default device: {devices[default_device]['name']}")
            return default_device
        
        try:
            device_id = int(selection)
            if device_id in input_devices:
                selected_device = devices[device_id]
                print(f"Selected: {selected_device['name']}")
                return device_id
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

# Select microphone before loading model
selected_device = select_microphone()

print("Loading Whisper model...")
model = whisper.load_model("base")  # use 'tiny' for speed

start_time = time.time()

def record_chunk(duration=CHUNK_DURATION, device=selected_device):
    print(f"Recording {duration}s from device {device}...")
    try:
        # Start recording
        audio = sd.rec(
            int(SAMPLE_RATE * duration), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='float32',
            device=device
        )
        
        # Wait with progress indicator
        import time
        start_time = time.time()
        for i in range(int(duration)):
            if i > 0:  # Don't sleep before showing first progress
                time.sleep(1)
            progress = '#' * (i+1) + '.' * (int(duration)-(i+1))
            print(f"\rRecording: [{progress}] {i+1}/{duration}s", end='')
            
            # If recording is taking too long, break out
            if time.time() - start_time > duration + 2:  # 2 sec grace period
                print("\nRecording is taking too long, continuing...")
                break
                
        print("\rRecording: [" + '#' * int(duration) + "] Complete!  ")
        
        # Wait for recording to complete with timeout
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"\nError during recording: {e}")
        # Return empty audio on error
        return np.zeros(int(SAMPLE_RATE * duration))

def get_wpm(transcript, duration_seconds):
    words = len(transcript.split())
    minutes = duration_seconds / 60
    return words / minutes if minutes > 0 else 0

# Enable debug output temporarily to diagnose the issue
DEBUG_VOLUME = True

def calculate_volume_stability(audio_data, segment_duration=0.5, plot_volumes=False):
    """
    Calculate volume stability by measuring the standard deviation of volume levels
    across audio segments. Returns a score from 0-100 where 100 is perfectly stable.
    
    Parameters:
    - audio_data: numpy array of audio samples
    - segment_duration: duration in seconds for each segment to analyze
    """
    # Skip if audio is too short
    if len(audio_data) < SAMPLE_RATE * segment_duration * 2:  # Need at least 2 segments
        if DEBUG_VOLUME:
            print("DEBUG: Audio too short for volume stability calculation")
        return 0
    
    # Calculate the number of samples per segment
    samples_per_segment = int(SAMPLE_RATE * segment_duration)
    
    # Split audio into segments
    num_segments = len(audio_data) // samples_per_segment
    segments = [audio_data[i*samples_per_segment:(i+1)*samples_per_segment] for i in range(num_segments)]
    
    # Filter out silent segments (often at the beginning or end)
    min_volume_threshold = 0.0001  # Reduced threshold to be more inclusive
    
    # Calculate RMS (volume) for each segment
    segment_volumes = []
    for segment in segments:
        # Apply a simple noise gate to remove background noise
        abs_segment = np.abs(segment)
        if np.mean(abs_segment) > min_volume_threshold:
            # Calculate RMS volume
            rms = np.sqrt(np.mean(np.square(segment)))
            segment_volumes.append(rms)
    
    # Print debug info
    if DEBUG_VOLUME:
        print(f"DEBUG: Found {len(segment_volumes)} non-silent segments out of {num_segments} total")
    
    # Skip if not enough valid segments
    if len(segment_volumes) < 2:  # Only need 2 segments minimum for calculation
        if DEBUG_VOLUME:
            print("DEBUG: Not enough non-silent segments for volume stability calculation")
        return 0
    
    # Calculate mean and standard deviation of volumes
    mean_volume = np.mean(segment_volumes)
    if mean_volume < 0.001:  # Lowered threshold for quiet audio
        if DEBUG_VOLUME:
            print("DEBUG: Mean volume too low for reliable calculation")
        return 0
        
    # Calculate coefficient of variation (normalized standard deviation)
    std_dev = np.std(segment_volumes)
    coefficient_of_variation = std_dev / mean_volume
    
    # Debug info
    if DEBUG_VOLUME:
        print(f"DEBUG: Volume stats - Mean: {mean_volume:.4f}, StdDev: {std_dev:.4f}, CV: {coefficient_of_variation:.4f}")
        
    # Optional plotting of volume levels for diagnostics
    if plot_volumes and len(segment_volumes) > 0:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(segment_volumes)
            plt.title(f"Volume Levels Over Time (CV: {coefficient_of_variation:.2f}, Score: {stability_score:.1f})")
            plt.xlabel("Segment Number (each {segment_duration}s)")
            plt.ylabel("RMS Volume")
            plt.axhline(y=mean_volume, color='r', linestyle='--', label=f"Mean: {mean_volume:.4f}")
            plt.legend()
            plt.grid(True)
            plt.savefig("volume_levels.png")
            print("Volume levels plot saved to 'volume_levels.png'")
        except Exception as e:
            print(f"Could not create volume plot: {e}")
            # Continue without plotting
    
    # Convert to stability score (0-100)
    # Lower variation = higher stability
    # Adjust the divisor to be more lenient - normal speech might have CV up to 0.7-1.0
    stability_score = 100 * max(0, min(1, 1 - (coefficient_of_variation / 1.0)))
    
    # Apply a floor to prevent extremely low scores when speaking
    if stability_score > 0:
        stability_score = max(20, stability_score)
    
    return stability_score

# while True:
#     chunk = record_chunk()
#     audio_buffer.append(chunk)

#     # Combine all audio from the beginning
#     full_audio = np.concatenate(audio_buffer)

#     # Save audio to a temp WAV file for Whisper
#     import tempfile
#     import soundfile as sf
#     temp_file = None
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#             temp_file = f.name
#             print("\nSaving audio to temporary file...")
#             sf.write(f.name, full_audio, SAMPLE_RATE)
            
#         print("Transcribing audio (this may take a few seconds)...")
#         # Simplified transcription without word timestamps to improve speed
#         result = model.transcribe(temp_file, language="en")
#         transcript = result["text"]
        
#         # Extract confidence scores for clarity calculation
#         segments = result.get("segments", [])
        
#         # Calculate average confidence across all segments
#         confidence_scores = [segment.get("avg_logprob", -10) for segment in segments if segment.get("avg_logprob") is not None]
        
#         # Store confidence data for clarity score calculation
#         if confidence_scores:
#             avg_confidence = sum(confidence_scores) / len(confidence_scores)
#             # Normalize to a 0-100 scale (logprobs are negative, with values closer to 0 being better)
#             # Typical values range from around -1 (very confident) to -2 or lower (less confident)
#             clarity_score = 100 * max(0, min(1, (avg_confidence + 3) / 3))
#         else:
#             # No segments with confidence scores
#             clarity_score = 0
#     except Exception as e:
#         print(f"Error during transcription: {str(e)}")
#         transcript = "[Transcription error]"
#         clarity_score = 0
#     finally:
#         # Clean up temp file
#         if temp_file and os.path.exists(temp_file):
#             try:
#                 os.unlink(temp_file)
#             except:
#                 pass

#     # Calculate WPM using total time
#     elapsed_time = time.time() - start_time
#     wpm = get_wpm(transcript, elapsed_time)
#     pace_score = 100 * math.exp(-((wpm - 80) / 45) ** 2)
    
#     # Calculate volume stability (set plot_volumes=True to generate diagnostic graph)
#     volume_stability = calculate_volume_stability(full_audio, segment_duration=0.5, plot_volumes=False)

#     print(f"\n[Time Elapsed: {int(elapsed_time)}s] Transcript so far: {transcript.strip()}")
#     print(f"WPM: {wpm:.2f} | Pacing Score: {pace_score:.1f}/100 | Clarity Score: {clarity_score:.1f}/100 | Volume Stability: {volume_stability:.1f}/100\n")
#     print(score_presentation_engagement(transcript.strip()))