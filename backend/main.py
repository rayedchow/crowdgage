# Fix OpenMP error first
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Fix multiprocessing issues
import multiprocessing
# Use spawn method instead of fork to prevent issues with threads
multiprocessing.set_start_method('spawn', force=True)

# Imports
import cv2
import numpy as np
import sounddevice as sd
import serial
import asyncio
import threading
import time
from datetime import datetime
import math
import sys
import json

# Project structure setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visual import ComprehensiveAnalyzer
import audio as audio_module

# Constants
SERIAL_PORT = '/dev/tty.usbmodem11401'
BAUD_RATE = 115200
SAMPLE_RATE = audio_module.SAMPLE_RATE

# Globals
ser = None
recording = False
stop_event = threading.Event()
cam = None
out = None
output_filename = None  # Track the current recording filename
resources_lock = threading.RLock()  # Lock for protecting resource access

# Buffers
frame_buffer = []
audio_buffer = []

# Score lists
clarity_scores = []
pace_scores = []
volume_scores = []
posture_scores = []
expression_scores = []
eyecontact_scores = []
speech_scores = []
engagement_scores = []
overall_scores = []

# Create recordings directory
os.makedirs("recordings", exist_ok=True)

# Analyzer
visual_analyzer = ComprehensiveAnalyzer()

# Video recording thread
def video_recording_loop():
    global cam, out, recording, stop_event, frame_buffer
    print("Video thread started")

    while recording and not stop_event.is_set():
        try:
            with resources_lock:
                if cam is None or not recording:
                    break
                
                ret, frame = cam.read()
                if not ret:
                    print("Frame capture failed")
                    continue

                if out is not None:
                    out.write(frame)
                
                # Make a copy of the frame to prevent memory issues
                frame_copy = frame.copy()
                frame_buffer.append(frame_copy)

                # Keep only the latest 10
                if len(frame_buffer) > 10:
                    # Clear references to old frames
                    del frame_buffer[0:len(frame_buffer)-10]
                    frame_buffer = frame_buffer[-10:]
        except Exception as e:
            print(f"Video thread error: {e}")
            
        time.sleep(1 / 30.0)

    print("Video thread stopped")

# Audio recording (1-second chunk)
async def record_audio_chunk(duration=1.0):
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=audio_module.selected_device)
    sd.wait()
    return audio_data[:, 0] if len(audio_data.shape) > 1 else audio_data

# Start recording
async def start_recording():
    global recording, cam, out, stop_event, frame_buffer, audio_buffer, output_filename
    global clarity_scores, pace_scores, volume_scores, posture_scores, expression_scores
    global eyecontact_scores, speech_scores, engagement_scores, overall_scores

    if recording:
        print("Already recording")
        await stop_recording()
        await asyncio.sleep(1)  # Give time for cleanup

    stop_event.clear()
    frame_buffer.clear()
    audio_buffer.clear()

    clarity_scores = [50]
    pace_scores = [50]
    volume_scores = [50]
    posture_scores = [50]
    expression_scores = [50]
    eyecontact_scores = [50]
    speech_scores = [50]
    engagement_scores = [50]
    overall_scores.clear()

    cam = cv2.VideoCapture(2)
    if not cam.isOpened():
        print("Error: Could not open camera")
        return

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"recordings/presentation_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (frame_width, frame_height))

    if not out.isOpened():
        print("Error initializing video writer")
        cam.release()
        return

    recording = True
    print(f"Recording started. Saving to {output_filename}")

    threading.Thread(target=video_recording_loop, daemon=True).start()
    asyncio.create_task(record_and_analyze())

    if ser:
        ser.write(b"RECORDING_STARTED\n")

# Stop recording
async def stop_recording():
    global recording, cam, out, output_filename

    if not recording:
        print("Not recording")
        return

    stop_event.set()
    recording = False
    
    # Allow video thread to exit
    await asyncio.sleep(0.5)
    
    with resources_lock:
        if cam:
            try:
                cam.release()
            except Exception as e:
                print(f"Error releasing camera: {e}")
            finally:
                cam = None
                
        if out:
            try:
                out.release()
            except Exception as e:
                print(f"Error releasing video writer: {e}")
            finally:
                out = None

    # Get the base filename from the current recording
    base_filename = output_filename.rsplit('.', 1)[0]  # Remove .mp4 extension
    json_filename = f"{base_filename}.json"

    # Prepare data for JSON export
    presentation_data = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": len(overall_scores),
        "scores": {
            "overall": overall_scores,
            "clarity": clarity_scores,
            "pace": pace_scores,
            "volume": volume_scores,
            "posture": posture_scores,
            "expression": expression_scores,
            "eye_contact": eyecontact_scores,
            "speech": speech_scores,
            "engagement": engagement_scores
        },
        "video_file": output_filename
    }

    # Save to JSON file
    try:
        with open(json_filename, 'w') as f:
            json.dump(presentation_data, f, indent=2)
        print(f"Presentation data saved to {json_filename}")
    except Exception as e:
        print(f"Error saving presentation data: {e}")

    print("Recording finished")
    print(f"\n===== RECORDING RESULTS =====\nDuration: {len(overall_scores)} seconds\nScores: {overall_scores}\n============================\n")

    if ser:
        ser.write(f"RECORDING_STOPPED\nDuration:{len(overall_scores)}s\nOverallScores:{','.join(map(str, overall_scores))}\n".encode('utf-8'))

# Main analysis loop
async def record_and_analyze():
    global clarity_scores, pace_scores, volume_scores, posture_scores, expression_scores
    global eyecontact_scores, speech_scores, engagement_scores, overall_scores, audio_buffer

    last_process_time = time.time()

    while recording and not stop_event.is_set():
        try:
            audio_chunk = await record_audio_chunk(1.0)
            audio_buffer.append(audio_chunk.copy())  # Create a copy to prevent memory issues

            current_time = time.time()
            if current_time - last_process_time >= 1.0 and frame_buffer:
                with resources_lock:
                    if not frame_buffer:
                        continue
                    # Make a copy to safely process outside the lock
                    latest_frame = frame_buffer[-1].copy()

                try:
                    processed_frame, scores = visual_analyzer.process_frame(latest_frame)
                    posture_scores.append(scores.get("posture_score", 50))
                    eyecontact_scores.append(scores.get("eye_contact_score", 50))
                    engagement_scores.append(scores.get("engagement_score", 50))
                    expression_scores.append(scores.get("focus_score", 50))
                except Exception as e:
                    print(f"Visual analysis error: {e}")
                    posture_scores.append(posture_scores[-1])
                    eyecontact_scores.append(eyecontact_scores[-1])
                    engagement_scores.append(engagement_scores[-1])
                    expression_scores.append(expression_scores[-1])

                try:
                    # Create a copy of the audio data to prevent memory issues
                    if len(audio_buffer) >= 3:
                        audio_to_process = np.concatenate([chunk.copy() for chunk in audio_buffer[-3:]])
                    else:
                        audio_to_process = audio_buffer[-1].copy()
                        
                    # Process audio in a try block to catch any transcription errors
                    result = audio_module.model.transcribe(audio_to_process)
                    transcript = result["text"].strip()
                    wpm = audio_module.get_wpm(transcript, len(audio_to_process) / SAMPLE_RATE)
                    volume_stability = audio_module.calculate_volume_stability(audio_to_process)

                    avg_logprob = 0
                    if 'segments' in result and result['segments']:
                        avg_logprob = sum(seg.get('avg_logprob', 0) for seg in result['segments']) / len(result['segments'])

                    clarity_score = int(max(0, min(100, (avg_logprob + 1) * 100)))
                    pacing_score = int(100 * math.exp(-((wpm - 80) / 45) ** 2))

                    try:
                        from text import score_presentation_engagement
                        speech_score = score_presentation_engagement(transcript)
                    except Exception as e:
                        print(f"Text scoring error: {e}")
                        speech_score = 50

                    clarity_scores.append(clarity_score)
                    pace_scores.append(pacing_score)
                    volume_scores.append(volume_stability)
                    speech_scores.append(speech_score)

                    print(f"Audio - WPM: {wpm:.1f}, Clarity: {clarity_score}, Pace: {pacing_score}, Volume: {volume_stability}")
                except Exception as e:
                    print(f"Audio error: {e}")
                    clarity_scores.append(clarity_scores[-1])
                    pace_scores.append(pace_scores[-1])
                    volume_scores.append(volume_scores[-1])
                    speech_scores.append(speech_scores[-1])

                latest = [
                    clarity_scores[-1],
                    pace_scores[-1],
                    volume_scores[-1],
                    posture_scores[-1],
                    expression_scores[-1],
                    eyecontact_scores[-1],
                    speech_scores[-1],
                    engagement_scores[-1]
                ]
                overall_score = int(sum(latest) / len(latest))
                overall_scores.append(overall_score)

                if ser:
                    ser.write(f"SCORE {overall_score}\n".encode('utf-8'))

                last_process_time = current_time

                if len(audio_buffer) > 10:
                    # Clear references to old buffers
                    del audio_buffer[0:len(audio_buffer)-10]
                    audio_buffer = audio_buffer[-10:]

            await asyncio.sleep(0.01)

        except Exception as e:
            print(f"Loop error: {e}")
            await asyncio.sleep(0.1)

    print("Analysis loop ended")

# Serial main loop
async def main():
    global ser
    print("Starting serial interface...")

    last_try = 0
    retry_delay = 5
    
    # Set up proper signal handlers
    try:
        import signal
        
        def cleanup_handler(signum, frame):
            print(f"Received signal {signum}, cleaning up...")
            stop_event.set()
            with resources_lock:
                if cam:
                    cam.release()
                if out:
                    out.release()
                if ser:
                    ser.close()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
    except Exception as e:
        print(f"Failed to set up signal handlers: {e}")

    while True:
        now = time.time()
        if not ser and now - last_try > retry_delay:
            try:
                print(f"Connecting to {SERIAL_PORT}...")
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                print("Serial connected.")
            except Exception as e:
                print(f"Serial error: {e}")
                ser = None
            last_try = now

        if ser:
            try:
                if ser.in_waiting > 0:
                    command = ser.readline().decode().strip().upper()
                    print(f"Command: {command}")
                    if command == "START RECORDING":
                        await start_recording()
                    elif command == "STOP RECORDING":
                        await stop_recording()
            except Exception as e:
                print(f"Serial read error: {e}")
                try: ser.close()
                except: pass
                ser = None

        await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Main loop exception: {e}")
    finally:
        # Final cleanup
        print("Performing final cleanup...")
        with resources_lock:
            if cam:
                cam.release()
            if out:
                out.release()
            if ser:
                ser.close()