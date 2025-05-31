"""
Simplified emotion detection for engagement and focus scoring.
Optimized for performance using preloaded model and faster backend.
"""
import cv2
import time
import numpy as np
from deepface import DeepFace
import threading

class SimpleEmotionDetector:
    def __init__(self):
        self.emotion = None
        self.processing = False
        self.last_frame = None
        self.emotion_thread = None
        self.detection_interval = 1.0
        self.last_detection_time = 0
        
        # Engagement and focus scores
        self.engagement_score = 50
        self.focus_score = 50
        
        # No model initialization - we'll use DeepFace.analyze directly
        # Emotion weights for engagement and focus
        self.engagement_weights = {
            'happy': 100,
            'surprise': 80,
            'neutral': 60,
            'sad': 40,
            'angry': 30,
            'fear': 30,
            'disgust': 20
        }

        self.focus_weights = {
            'neutral': 100,
            'surprise': 90,
            'happy': 70,
            'angry': 60,
            'sad': 50,
            'fear': 40,
            'disgust': 30
        }

    def process_emotion_async(self, frame):
        """Process emotion in a separate thread"""
        if self.processing:
            return

        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return

        self.processing = True
        self.last_frame = frame.copy()

        self.emotion_thread = threading.Thread(target=self._detect_emotion)
        self.emotion_thread.daemon = True
        self.emotion_thread.start()

    def _detect_emotion(self):
        """Detect emotion in a separate thread"""
        try:
            # Resize frame to improve performance
            small_frame = cv2.resize(self.last_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Analyze with DeepFace - use a simpler approach
            result = DeepFace.analyze(
                img_path=small_frame,  # Use img_path parameter name
                actions=['emotion'],  
                enforce_detection=False,  # Don't error if face not detected
                silent=True  # Suppress output to console
            )
            
            # Handle result format, which could be dictionary or list depending on version
            if isinstance(result, dict):
                # Newer DeepFace versions return a dict
                emotion_data = result
                dominant_emotion = emotion_data.get('dominant_emotion')
            elif isinstance(result, list) and len(result) > 0:
                # Older DeepFace versions return a list
                emotion_data = result[0]
                dominant_emotion = emotion_data.get('dominant_emotion')
            else:
                print("No valid emotion data detected")
                self.processing = False
                self.last_detection_time = time.time()
                return
                
            if dominant_emotion:
                # Update scores
                self.engagement_score = self.engagement_weights.get(dominant_emotion, 50)
                self.focus_score = self.focus_weights.get(dominant_emotion, 50)
                
                self.emotion = dominant_emotion
                print(f"Detected emotion: {dominant_emotion}")
            else:
                print("No face detected")
                
        except Exception as e:
            print(f"Emotion detection error: {e}")
        finally:
            self.processing = False
            self.last_detection_time = time.time()
            print("Finished emotion detection.")

    def get_status_text(self, score):
        if score >= 80:
            return "Excellent", (0, 255, 0)
        elif score >= 60:
            return "Good", (0, 255, 255)
        elif score >= 40:
            return "Fair", (0, 165, 255)
        else:
            return "Poor", (0, 0, 255)

    def draw_feedback(self, frame):
        if self.emotion:
            engagement_status, engagement_color = self.get_status_text(self.engagement_score)
            focus_status, focus_color = self.get_status_text(self.focus_score)

            cv2.putText(frame, f"Emotion: {self.emotion.capitalize()}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"Engagement: {self.engagement_score}/100 ({engagement_status})", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, engagement_color, 2, cv2.LINE_AA)

            cv2.putText(frame, f"Focus: {self.focus_score}/100 ({focus_status})", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, focus_color, 2, cv2.LINE_AA)

        return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    detector = SimpleEmotionDetector()

    print("Starting emotion analysis...")
    print("Press 'q' to quit")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        detector.process_emotion_async(frame)
        frame = detector.draw_feedback(frame)
        cv2.imshow("Emotion Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()