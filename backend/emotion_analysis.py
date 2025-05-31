"""
Standalone emotion detection script for engagement and focus scoring.
This script does not depend on mediapipe and can be run independently.
"""
import cv2
import time
import numpy as np
from deepface import DeepFace
from emotion_detector import EmotionDetector

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize emotion detector
    emotion_detector = EmotionDetector()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    
    # Create video writer (optional)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_output = cv2.VideoWriter('emotion_output.mp4', fourcc, fps, frame_size)
    
    print("Starting emotion analysis...")
    print("Press 'q' to quit")
    
    # Main loop
    while True:
        # Read frame
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
            
        # Process frame with emotion detector
        emotion_data = emotion_detector.detect_emotion(frame)
        
        if emotion_data:
            # Calculate scores
            engagement_score = emotion_detector.calculate_engagement_score()
            focus_score = emotion_detector.calculate_focus_score()
            
            # Display information about scores
            print(f"Emotion: {emotion_data['dominant_emotion']} | " +
                  f"Engagement: {engagement_score}/100 | " +
                  f"Focus: {focus_score}/100")
                  
        # Draw feedback on frame
        frame = emotion_detector.draw_emotion_feedback(frame, emotion_data)
        
        # Show frame
        cv2.imshow("Emotion Analysis", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    # video_output.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
