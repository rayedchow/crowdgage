"""
Emotion detection module using deepface to score user engagement and focus.
"""
import cv2
from deepface import DeepFace
import numpy as np
import time

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detector with focus and engagement scoring parameters."""
        self.last_detection_time = 0
        self.detection_interval = 1.0  # Analyze every 1 second to reduce CPU load
        self.last_emotion = None
        self.emotion_history = []
        self.max_history = 10  # Keep last 10 emotion readings for smoothing
        
        # Define which emotions contribute positively to engagement
        self.engagement_emotions = {
            'happy': 1.0,       # Positive engagement
            'surprise': 0.8,    # Often indicates interest
            'neutral': 0.6,     # Baseline attention
            'sad': 0.4,         # Lower engagement
            'angry': 0.3,       # Might indicate frustration with content
            'fear': 0.3,        # Might indicate anxiety
            'disgust': 0.2      # Typically negative engagement
        }
        
        # Define which emotions contribute positively to focus
        self.focus_emotions = {
            'neutral': 1.0,     # Strong focus is often neutral-faced
            'surprise': 0.9,    # Can indicate concentration
            'happy': 0.7,       # Positive but may be distracted
            'angry': 0.6,       # Might be focused but negatively
            'sad': 0.5,         # May indicate lower focus
            'fear': 0.4,        # Might be distracted by anxiety
            'disgust': 0.3      # Typically indicates distraction
        }

    def detect_emotion(self, frame):
        """
        Detect emotion in the given frame using deepface.
        
        Args:
            frame: The image frame to analyze
            
        Returns:
            dict: Detected emotion information or None if detection failed
        """
        current_time = time.time()
        
        # Only run detection at specified intervals to reduce CPU load
        if current_time - self.last_detection_time < self.detection_interval:
            return self.last_emotion
            
        # Make a copy of the frame to avoid modifying the original
        frame_copy = frame.copy()
        
        # Resize the frame to make processing faster
        # This can significantly improve performance
        frame_small = cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5)
        
        try:
            # Use a simplified approach for emotion detection
            # Setting detector_backend to 'opencv' can be faster
            result = DeepFace.analyze(
                frame_small, 
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,  # Don't error if face not detected
                silent=True,  # Suppress console output
                prog_bar=False  # Don't show progress bar
            )
            
            if isinstance(result, list) and len(result) > 0:
                emotion_data = result[0]
                dominant_emotion = emotion_data['dominant_emotion']
                emotion_scores = emotion_data['emotion']
                
                self.last_emotion = {
                    'dominant_emotion': dominant_emotion,
                    'emotion_scores': emotion_scores
                }
                
                # Update emotion history for smoothing
                self.emotion_history.append(self.last_emotion)
                if len(self.emotion_history) > self.max_history:
                    self.emotion_history.pop(0)
                    
                self.last_detection_time = current_time
                print(f"Detected emotion: {dominant_emotion}")
                return self.last_emotion
            else:
                print("No face detected in frame")
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            
        return self.last_emotion  # Return last known emotion instead of None to maintain continuity
        
    def calculate_engagement_score(self):
        """
        Calculate an engagement score (0-100) based on detected emotions.
        
        Returns:
            int: Engagement score from 0-100
        """
        if not self.emotion_history:
            return 50  # Default neutral score if no data
            
        # Use weighted average of recent emotions for smoother scoring
        weights = np.linspace(0.5, 1.0, len(self.emotion_history))
        weights = weights / np.sum(weights)  # Normalize weights
        
        engagement_score = 0
        for i, emotion_data in enumerate(self.emotion_history):
            emotion_scores = emotion_data['emotion_scores']
            # Calculate weighted contribution of each emotion
            emotion_engagement = sum(
                self.engagement_emotions.get(emotion, 0.5) * score / 100
                for emotion, score in emotion_scores.items()
            )
            engagement_score += weights[i] * emotion_engagement
            
        # Convert to 0-100 scale
        engagement_score = int(engagement_score * 100)
        return min(100, max(0, engagement_score))
        
    def calculate_focus_score(self):
        """
        Calculate a focus score (0-100) based on detected emotions.
        
        Returns:
            int: Focus score from 0-100
        """
        if not self.emotion_history:
            return 50  # Default neutral score if no data
            
        # Use weighted average of recent emotions for smoother scoring
        weights = np.linspace(0.5, 1.0, len(self.emotion_history))
        weights = weights / np.sum(weights)  # Normalize weights
        
        focus_score = 0
        for i, emotion_data in enumerate(self.emotion_history):
            emotion_scores = emotion_data['emotion_scores']
            # Calculate weighted contribution of each emotion
            emotion_focus = sum(
                self.focus_emotions.get(emotion, 0.5) * score / 100
                for emotion, score in emotion_scores.items()
            )
            focus_score += weights[i] * emotion_focus
            
        # Convert to 0-100 scale
        focus_score = int(focus_score * 100)
        return min(100, max(0, focus_score))
    
    def get_engagement_status(self, score):
        """
        Get a textual status based on engagement score.
        
        Args:
            score: Engagement score (0-100)
            
        Returns:
            tuple: (status_text, color)
        """
        if score >= 80:
            return "Highly Engaged", (0, 255, 0)  # Green
        elif score >= 60:
            return "Engaged", (0, 255, 255)  # Yellow
        elif score >= 40:
            return "Moderately Engaged", (0, 165, 255)  # Orange
        else:
            return "Disengaged", (0, 0, 255)  # Red
            
    def get_focus_status(self, score):
        """
        Get a textual status based on focus score.
        
        Args:
            score: Focus score (0-100)
            
        Returns:
            tuple: (status_text, color)
        """
        if score >= 80:
            return "Highly Focused", (0, 255, 0)  # Green
        elif score >= 60:
            return "Focused", (0, 255, 255)  # Yellow
        elif score >= 40:
            return "Moderately Focused", (0, 165, 255)  # Orange
        else:
            return "Distracted", (0, 0, 255)  # Red
    
    def draw_emotion_feedback(self, image, emotion_data=None):
        """
        Draw emotion, engagement and focus feedback on the image.
        
        Args:
            image: The image to draw on
            emotion_data: Optional emotion data if already detected
            
        Returns:
            The image with drawn feedback
        """
        if emotion_data is None:
            emotion_data = self.last_emotion
            
        h, w = image.shape[:2]
        
        if emotion_data:
            # Get scores
            engagement_score = self.calculate_engagement_score()
            focus_score = self.calculate_focus_score()
            
            # Get status text and colors
            engagement_status, engagement_color = self.get_engagement_status(engagement_score)
            focus_status, focus_color = self.get_focus_status(focus_score)
            
            # Draw dominant emotion
            cv2.putText(
                image,
                f"Emotion: {emotion_data['dominant_emotion'].capitalize()}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            # Draw engagement score
            cv2.putText(
                image,
                f"Engagement: {engagement_score}/100 ({engagement_status})",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                engagement_color,
                2,
                cv2.LINE_AA,
            )
            
            # Draw focus score
            cv2.putText(
                image,
                f"Focus: {focus_score}/100 ({focus_status})",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                focus_color,
                2,
                cv2.LINE_AA,
            )
        
        return image
