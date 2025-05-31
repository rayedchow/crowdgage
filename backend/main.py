# Fix OpenMP error first
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import essential libraries
import os
from datetime import datetime
import asyncio
import threading
import time
import cv2
import numpy as np
import sounddevice as sd
import serial
import math  # Required for pacing score calculation

# Import modules from existing code
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visual import ComprehensiveAnalyzer
import audio as audio_module  # Import as module to avoid name conflicts

# Set OpenMP environment variable to avoid errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Serial port configuration
SERIAL_PORT = '/dev/tty.usbmodem112201'  # or COM3, etc.
BAUD_RATE = 115200

# Initialize serial variable, we'll attempt connection in the main loop
ser = None
print(f"Will attempt to connect to serial port {SERIAL_PORT} at {BAUD_RATE} baud")

# Global variables
recording = False
last_processed_time = 0
SAMPLE_RATE = audio_module.SAMPLE_RATE  # Use sample rate from audio.py
video_capture = None
video_writer = None
stop_event = threading.Event()

# Create output directory if it doesn't exist
os.makedirs('recordings', exist_ok=True)

# Score lists that track all measurements
clarity_scores = []
pace_scores = []
volume_scores = []
posture_scores = []
expression_scores = []
eyecontact_scores = []
speech_scores = []
engagement_scores = []

# Overall scores (calculated each second)
overall_scores = []

# Buffers for processing
frame_buffer = []
audio_buffer = []

# Initialize analyzers
visual_analyzer = ComprehensiveAnalyzer()  # Same analyzer used in visual.py



# Record audio asynchronously
async def record_audio_chunk(duration=1.0):
    """
    Record audio chunk using sounddevice
    """
    # Use the sample rate from audio.py
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=1,
        dtype='float32',
        device=audio_module.selected_device
    )
    
    # Wait for recording to complete
    sd.wait()
    
    # Ensure audio data is a numpy array
    if not isinstance(audio_data, np.ndarray):
        raise ValueError(f"Unexpected audio data type: {type(audio_data)}")
        
    # Ensure 2D shape and convert to 1D if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Take first channel only
    
    return audio_data

# Function to start recording
async def start_recording():
    """Start recording and analyzing audio/video"""
    global recording, video_capture, frame_buffer, audio_buffer, stop_event, video_writer, frame_count, output_filename
    global clarity_scores, pace_scores, volume_scores, posture_scores, expression_scores
    global eyecontact_scores, speech_scores, engagement_scores, overall_scores
    
    if recording:
        print("Already recording")
        return
    
    # Reset buffers and state
    stop_event.clear()
    frame_buffer = []
    audio_buffer = []
    
    # Reset score lists
    clarity_scores = [50]  # Start with default scores
    pace_scores = [50]
    volume_scores = [50]
    posture_scores = [50]
    expression_scores = [50]
    eyecontact_scores = [50]
    speech_scores = [50]
    engagement_scores = [50]
    overall_scores = []
    
    # Initialize camera
    video_capture = cv2.VideoCapture(1)  # Use second camera (index 1) as requested
    if not video_capture.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create video writer for MP4 output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"recordings/presentation_{timestamp}.mp4"
    frame_count = 0

    # Get camera properties for video writer
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0  # Target FPS for the output video
    
    print(f"Camera resolution: {frame_width}x{frame_height}")

    # Initialize the video writer with explicit parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height), True)
    
    if not video_writer.isOpened():
        print("Error: Could not initialize video writer")
        return

    # Start recording
    recording = True
    
    # Reset frame count
    frame_count = 0
    
    # Set a higher priority for video recording
    print(f"Starting recording... Saving to {output_filename}")
    # Create a dedicated task for recording with higher priority
    recording_task = asyncio.create_task(record_and_analyze())
    
    # Send confirmation back through serial port
    if ser:
        ser.write(b"RECORDING_STARTED\n")

# Function to stop recording
async def stop_recording():
    """Stop recording and print scores"""
    global recording, video_capture, stop_event, video_writer, frame_count, output_filename
    
    if not recording:
        print("Not recording")
        return
    
    # Signal threads to stop
    stop_event.set()
    recording = False
    
    # Clean up resources
    if video_capture:
        video_capture.release()

    # Finalize and release the video writer
    if video_writer and video_writer.isOpened():
        # Make sure all frames are flushed to disk
        video_writer.release()
        print(f"Video saved to {output_filename} with {frame_count} frames")
        if frame_count > 0:
            print(f"Video duration approximately {frame_count/30:.2f} seconds")
        else:
            print("Warning: No frames were recorded!")
    else:
        print("Warning: Video writer was not properly opened or already released")
    
    print("\n===== RECORDING RESULTS =====\n")
    print(f"Total duration: {len(overall_scores)} seconds")
    print(f"Overall scores per second: {overall_scores}")
    print("\n============================\n")
    
    # Send results back through serial port
    if ser:
        result_message = f"RECORDING_STOPPED\nDuration:{len(overall_scores)}s\nOverallScores:{','.join(map(str, overall_scores))}\n"
        ser.write(result_message.encode('utf-8'))
        
        # Also send detailed scores
        detailed_scores = f"ClarityScores:{','.join(map(str, clarity_scores))}\n"
        detailed_scores += f"PaceScores:{','.join(map(str, pace_scores))}\n"
        detailed_scores += f"VolumeScores:{','.join(map(str, volume_scores))}\n"
        detailed_scores += f"PostureScores:{','.join(map(str, posture_scores))}\n"
        detailed_scores += f"ExpressionScores:{','.join(map(str, expression_scores))}\n"
        detailed_scores += f"EyeContactScores:{','.join(map(str, eyecontact_scores))}\n"
        detailed_scores += f"SpeechScores:{','.join(map(str, speech_scores))}\n"
        detailed_scores += f"EngagementScores:{','.join(map(str, engagement_scores))}\n"
        ser.write(detailed_scores.encode('utf-8'))

# Variables to track video recording
frame_count = 0
output_filename = ""

async def record_and_analyze():
    """Main function to record and analyze audio and video"""
    global clarity_scores, pace_scores, volume_scores, posture_scores, expression_scores, eyecontact_scores, speech_scores, engagement_scores, overall_scores, video_writer, frame_count
    
    last_process_time = time.time()
    frame_buffer = []
    audio_buffer = []
    frame_count = 0
    
    print("Starting recording and analysis...")
    
    while recording and not stop_event.is_set():
        try:
            # Capture frame from camera
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame")
                await asyncio.sleep(0.1)
                continue
            
            # Make a copy of the frame to ensure it's not modified before writing
            frame_to_write = frame.copy()
            
            # Write frame to video file and increment counter
            if video_writer and video_writer.isOpened():
                video_writer.write(frame_to_write)
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames (approximately 1 second at 30fps)
                    print(f"Recorded {frame_count} frames so far")
            else:
                print("Warning: Video writer not available or not opened")

            # Record audio chunk (1 second)
            audio_chunk = await record_audio_chunk(1.0)
            
            # Add to local buffers
            frame_buffer.append(frame)
            audio_buffer.append(audio_chunk)
            
            # Process at 1-second intervals
            current_time = time.time()
            if current_time - last_process_time >= 1.0:
                # Process the latest frame with visual analyzer
                if frame_buffer:
                    latest_frame = frame_buffer[-1]
                    try:
                        # Process with visual analyzer
                        processed_frame, scores = visual_analyzer.process_frame(latest_frame)
                        
                        # Extract scores from visual analysis
                        posture_score = scores.get("posture_score", 50)
                        eye_contact_score = scores.get("eye_contact_score", 50)
                        engagement_score = scores.get("engagement_score", 50)
                        expression_score = scores.get("focus_score", 50)  # Using focus as expression
                        
                        # Append to score lists
                        posture_scores.append(posture_score)
                        eyecontact_scores.append(eye_contact_score)
                        engagement_scores.append(engagement_score)
                        expression_scores.append(expression_score)
                        
                        print(f"Visual scores - Posture: {posture_score}, Eye Contact: {eye_contact_score}")
                    except Exception as e:
                        print(f"Visual analysis error: {str(e)}")
                        # Append last values or defaults if available
                        if posture_scores:
                            posture_scores.append(posture_scores[-1])
                        else:
                            posture_scores.append(50)
                            
                        if eyecontact_scores:
                            eyecontact_scores.append(eyecontact_scores[-1])
                        else:
                            eyecontact_scores.append(50)
                            
                        if engagement_scores:
                            engagement_scores.append(engagement_scores[-1])
                        else:
                            engagement_scores.append(50)
                            
                        if expression_scores:
                            expression_scores.append(expression_scores[-1])
                        else:
                            expression_scores.append(50)
                
                # Process audio for speech analysis
                if audio_buffer:
                    try:
                        # Combine recent chunks for better analysis
                        combined_audio = np.concatenate(audio_buffer[-3:]) if len(audio_buffer) >= 3 else audio_buffer[-1]
                        
                        # Transcribe using Whisper
                        result = audio_module.model.transcribe(combined_audio)
                        transcript = result["text"].strip()
                        
                        # Calculate metrics
                        chunk_duration = len(combined_audio) / SAMPLE_RATE
                        wpm = audio_module.get_wpm(transcript, chunk_duration)
                        volume_stability = audio_module.calculate_volume_stability(combined_audio)
                        
                        # Extract clarity score from Whisper confidence
                        avg_logprob = 0
                        if 'segments' in result and result['segments']:
                            total_prob = sum(seg.get('avg_logprob', 0) for seg in result['segments'])
                            avg_logprob = total_prob / len(result['segments'])
                        
                        # Normalize to 0-100 scale
                        clarity_score = int(max(0, min(100, (avg_logprob + 1) * 100)))
                        
                        # Calculate pacing score (ideal around 80 WPM)
                        pacing_score = int(100 * math.exp(-((wpm - 80) / 45) ** 2))
                        
                        # Use text.py to get speech and overall engagement
                        speech_score = 0
                        try:
                            from text import score_presentation_engagement
                            speech_score = score_presentation_engagement(transcript)
                        except Exception as text_err:
                            print(f"Text analysis error: {str(text_err)}")
                            speech_score = 50  # Default value
                        
                        # Append to score lists
                        clarity_scores.append(clarity_score)
                        pace_scores.append(pacing_score)
                        volume_scores.append(volume_stability)
                        speech_scores.append(speech_score)
                        
                        print(f"Audio scores - Clarity: {clarity_score}, Pace: {pacing_score}, Volume: {volume_stability}")
                        print(f"Transcript: {transcript[:50]}..." if len(transcript) > 50 else f"Transcript: {transcript}")
                    except Exception as e:
                        print(f"Audio analysis error: {str(e)}")
                        # Append last values or defaults if available
                        if clarity_scores:
                            clarity_scores.append(clarity_scores[-1])
                        else:
                            clarity_scores.append(50)
                            
                        if pace_scores:
                            pace_scores.append(pace_scores[-1])
                        else:
                            pace_scores.append(50)
                            
                        if volume_scores:
                            volume_scores.append(volume_scores[-1])
                        else:
                            volume_scores.append(50)
                            
                        if speech_scores:
                            speech_scores.append(speech_scores[-1])
                        else:
                            speech_scores.append(50)
                
                # Calculate overall score for this second
                try:
                    # Get latest scores (use the most recent value from each category)
                    latest_scores = [
                        clarity_scores[-1] if clarity_scores else 50,
                        pace_scores[-1] if pace_scores else 50,
                        volume_scores[-1] if volume_scores else 50,
                        posture_scores[-1] if posture_scores else 50,
                        expression_scores[-1] if expression_scores else 50,
                        eyecontact_scores[-1] if eyecontact_scores else 50,
                        speech_scores[-1] if speech_scores else 50,
                        engagement_scores[-1] if engagement_scores else 50
                    ]
                    
                    # Calculate overall score as mean of all scores
                    overall_score = int(sum(latest_scores) / len(latest_scores))

                    overall_scores.append(overall_score)
                    
                    print(f"Overall score at {len(overall_scores)}s: {overall_score}")
                    
                    # Send overall score to serial port
                    if ser:
                        score_message = f"SCORE {overall_score}\n"
                        ser.write(score_message.encode('utf-8'))
                except Exception as e:
                    print(f"Error calculating overall score: {str(e)}")
                    # Use previous overall score or default
                    if overall_scores:
                        overall_scores.append(overall_scores[-1])
                    else:
                        overall_scores.append(50)
                
                # Update timestamp
                last_process_time = current_time
                
                # Clean up buffers (keep only recent chunks)
                if len(frame_buffer) > 10:
                    frame_buffer = frame_buffer[-10:]
                if len(audio_buffer) > 10:
                    audio_buffer = audio_buffer[-10:]
            
            # Short sleep to avoid CPU overload
            await asyncio.sleep(0.01)
            
        except Exception as e:
            print(f"Error in record_and_analyze: {str(e)}")
            await asyncio.sleep(0.1)  # Sleep on error
    
    print("Recording and analysis stopped")
    
# Main serial communication loop
async def main():
    global ser
    
    print("Starting CrowdGage serial communication...")
    print(f"Listening for START RECORDING and STOP RECORDING commands")
    
    # Initialize hardware
    print("Initializing analyzers...")
    
    # Main program loop
    connection_retry_interval = 5  # seconds
    last_connection_attempt = 0
    
    while True:
        # Try to connect to serial port if not connected
        current_time = time.time()
        if not ser and (current_time - last_connection_attempt > connection_retry_interval):
            try:
                print(f"Attempting to connect to serial port {SERIAL_PORT}...")
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                print(f"Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud")
            except Exception as e:
                print(f"Connection failed: {str(e)}")
                ser = None
            last_connection_attempt = current_time
        
        # Process serial commands if connected
        if ser:
            try:
                if ser.in_waiting > 0:
                    command = ser.readline().decode('utf-8').strip().upper()
                    print(f"Received command: {command}")
                    
                    if command == "START RECORDING":
                        await start_recording()
                    elif command == "STOP RECORDING":
                        await stop_recording()
                    else:
                        print(f"Unknown command: {command}")
            except Exception as e:
                print(f"Serial error: {str(e)}")
                # If there's an error with the serial port, close it and set to None to attempt reconnection
                try:
                    ser.close()
                except:
                    pass
                ser = None
                print("Serial connection lost. Will attempt to reconnect.")
        
        # Small delay to prevent CPU hogging
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(main())
