"""
Comprehensive analysis combining posture tracking and emotion detection.
Provides a unified interface for posture score and engagement/focus metrics.
"""
import cv2
import time
import math as m
import numpy as np
import threading
import mediapipe as mp
from deepface import DeepFace
import collections

class PostureAnalyzer:
    """Analyzes posture using MediaPipe pose detection"""
    
    def __init__(self):
        """Initialize posture analyzer"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.good_frames = 0
        self.bad_frames = 0
        self.last_posture_score = 50  # Default middle score
    
    def find_distance(self, x1, y1, x2, y2):
        """Calculate the distance between two points"""
        dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist
    
    def find_angle(self, x1, y1, x2, y2):
        """Calculate the angle between two points and the horizontal"""
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / m.pi) * theta
        return degree
    
    def calculate_posture_score(self, neck_inclination, torso_inclination):
        """
        Calculate a posture score from 0-100 based on neck and torso angles.
        
        Args:
            neck_inclination: The neck angle in degrees.
            torso_inclination: The torso angle in degrees.
            
        Returns:
            A posture score from 0 to 100 as an integer.
        """
        # Ideal ranges
        ideal_neck_range = (35, 45)
        ideal_torso_range = (0, 10)
        
        # Calculate neck score (60% weight)
        if neck_inclination < ideal_neck_range[0]:
            # Too far forward
            neck_score = max(0, 100 - (ideal_neck_range[0] - neck_inclination) * 3)
        elif neck_inclination > ideal_neck_range[1]:
            # Too far back
            neck_score = max(0, 100 - (neck_inclination - ideal_neck_range[1]) * 3)
        else:
            # Within ideal range
            neck_score = 100
        
        # Calculate torso score (40% weight)
        if torso_inclination < ideal_torso_range[0]:
            # Too far forward
            torso_score = max(0, 100 - (ideal_torso_range[0] - torso_inclination) * 5)
        elif torso_inclination > ideal_torso_range[1]:
            # Too far back
            torso_score = max(0, 100 - (torso_inclination - ideal_torso_range[1]) * 5)
        else:
            # Within ideal range
            torso_score = 100
        
        # Calculate weighted score
        posture_score = int(0.6 * neck_score + 0.4 * torso_score)
        self.last_posture_score = posture_score
        
        return posture_score
    
    def analyze_posture(self, frame):
        """
        Analyze posture in the given frame.
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            tuple: (processed_frame, posture_score)
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = self.pose.process(image_rgb)
        
        if not keypoints.pose_landmarks:
            # No pose detected
            cv2.putText(
                frame,
                "No pose detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
            return frame, self.last_posture_score
        
        # Extract key landmarks
        lm = keypoints.pose_landmarks
        lmPose = self.mp_pose.PoseLandmark
        h, w = frame.shape[:2]
        
        try:
            # Extract key points
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
            
            # Calculate shoulder alignment
            offset = self.find_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            
            # Calculate neck and torso angles
            neck_inclination = self.find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = self.find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            
            # Calculate posture score
            posture_score = self.calculate_posture_score(neck_inclination, torso_inclination)
            
            # Update frame counters
            if posture_score >= 60:  # Good posture threshold
                self.bad_frames = 0
                self.good_frames += 1
                color = (127, 233, 100)  # Light green
            else:
                self.good_frames = 0
                self.bad_frames += 1
                color = (0, 0, 255)  # Red
            
            # Draw angles on frame
            angle_text = f'Neck: {int(neck_inclination)}°  Torso: {int(torso_inclination)}°'
            cv2.putText(frame, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw landmarks and connections
            cv2.circle(frame, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)  # Yellow
            cv2.circle(frame, (l_ear_x, l_ear_y), 7, (0, 255, 255), -1)
            cv2.circle(frame, (r_shldr_x, r_shldr_y), 7, (255, 0, 255), -1)  # Pink
            cv2.circle(frame, (l_hip_x, l_hip_y), 7, (0, 255, 255), -1)
            
            # Draw connecting lines
            cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 3)
            cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 3)
            
            return frame, posture_score
            
        except Exception as e:
            print(f"Posture analysis error: {e}")
            return frame, self.last_posture_score


class EyeContactAnalyzer:
    """Analyzes eye contact using MediaPipe face mesh"""
    
    def __init__(self):
        """Initialize eye contact analyzer"""
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye contact tracking
        self.eye_contact_score = 50  # Default middle score
        self.gaze_history = collections.deque(maxlen=30)  # Store recent gaze directions
        self.looking_at_camera = False
        self.frames_with_face = 0
        self.last_detection_time = 0
        
        # Use basic eye landmarks instead of iris (more reliable)
        # Left eye landmarks
        self.LEFT_EYE = [
            33, 160, 158, 133, 153, 144,  # upper lid
            163, 7, 173, 246, 249, 390     # lower lid
        ]
        # Right eye landmarks
        self.RIGHT_EYE = [
            362, 382, 381, 380, 374, 373,  # upper lid
            390, 249, 263, 466, 388, 387   # lower lid
        ]
        
        # Center of eyes
        self.LEFT_EYE_CENTER = 473  # approximate center of left eye
        self.RIGHT_EYE_CENTER = 468  # approximate center of right eye
        
        # Face orientation landmarks
        self.NOSE_TIP = 4
        self.CHIN = 152
        self.LEFT_EYE_LEFT = 33
        self.LEFT_EYE_RIGHT = 133
        self.RIGHT_EYE_LEFT = 362
        self.RIGHT_EYE_RIGHT = 263
    
    def analyze_eye_contact(self, frame):
        """Analyze eye contact in the given frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            # If no face found for a while, gradually reduce score
            if time.time() - self.last_detection_time > 3:
                self.eye_contact_score = max(0, self.eye_contact_score - 5)
            return frame, self.eye_contact_score
        
        # Face found, update detection time
        self.last_detection_time = time.time()
        self.frames_with_face += 1
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        try:
            # Get face orientation landmarks
            nose_tip = np.array([face_landmarks.landmark[self.NOSE_TIP].x, 
                               face_landmarks.landmark[self.NOSE_TIP].y, 
                               face_landmarks.landmark[self.NOSE_TIP].z])
            
            # Left eye center
            left_eye_pts = []
            for lm_idx in self.LEFT_EYE:
                x = face_landmarks.landmark[lm_idx].x * w
                y = face_landmarks.landmark[lm_idx].y * h
                left_eye_pts.append((int(x), int(y)))
            
            # Right eye center
            right_eye_pts = []
            for lm_idx in self.RIGHT_EYE:
                x = face_landmarks.landmark[lm_idx].x * w
                y = face_landmarks.landmark[lm_idx].y * h
                right_eye_pts.append((int(x), int(y)))
            
            # Create eye hull for visualization
            left_eye_hull = cv2.convexHull(np.array(left_eye_pts))
            right_eye_hull = cv2.convexHull(np.array(right_eye_pts))
            
            # Calculate head pose estimation (simplified)
            # Check if face is relatively front-facing
            # Use Z-coordinate of nose tip (negative when facing camera)
            facing_camera = nose_tip[2] < 0
            
            # Calculate eye centers
            left_eye_center = (int(face_landmarks.landmark[self.LEFT_EYE_CENTER].x * w),
                             int(face_landmarks.landmark[self.LEFT_EYE_CENTER].y * h))
            right_eye_center = (int(face_landmarks.landmark[self.RIGHT_EYE_CENTER].x * w),
                              int(face_landmarks.landmark[self.RIGHT_EYE_CENTER].y * h))
            
            # Calculate head yaw (left-right rotation)
            # In a front-facing position, eyes should be approximately horizontally aligned
            eye_level_diff = abs(left_eye_center[1] - right_eye_center[1])
            eye_level_threshold = 0.03 * h  # 3% of image height
            eyes_level = eye_level_diff < eye_level_threshold
            
            # Determine if making eye contact based on multiple factors
            looking_at_camera = facing_camera and eyes_level
            
            # Track history of eye contact
            self.gaze_history.append(1 if looking_at_camera else 0)
            
            # Calculate eye contact score based on recent history
            if len(self.gaze_history) > 0:
                eye_contact_percentage = sum(self.gaze_history) / len(self.gaze_history)
                # Gradually adjust score for smoother transitions
                target_score = int(eye_contact_percentage * 100)
                self.eye_contact_score = int(0.8 * self.eye_contact_score + 0.2 * target_score)
            
            # Ensure score is within valid range
            self.eye_contact_score = max(0, min(100, self.eye_contact_score))
            
            # Visualize eye contact
            eye_color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
            cv2.drawContours(frame, [left_eye_hull], -1, eye_color, 1)
            cv2.drawContours(frame, [right_eye_hull], -1, eye_color, 1)
            cv2.circle(frame, left_eye_center, 3, eye_color, -1)
            cv2.circle(frame, right_eye_center, 3, eye_color, -1)
            
            # Add debug info
            cv2.putText(frame, f"Face: {'Front' if facing_camera else 'Side'}", 
                      (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Eyes Level: {'Yes' if eyes_level else 'No'}", 
                      (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame, self.eye_contact_score
            
        except Exception as e:
            print(f"Eye contact analysis error: {e}")
            # If error occurs occasionally, still increment score a bit if we've been detecting face
            if self.frames_with_face > 10:
                self.eye_contact_score = min(100, self.eye_contact_score + 5)
            return frame, self.eye_contact_score


class EmotionAnalyzer:
    """Analyzes emotions using DeepFace to determine engagement and focus"""
    
    def __init__(self):
        """Initialize emotion analyzer"""
        self.emotion = "neutral"  # Default emotion
        self.emotion_scores = {}
        self.processing = False
        self.emotion_thread = None
        self.detection_interval = 2.0  # seconds
        self.last_detection_time = 0
        
        # Default scores
        self.engagement_score = 50
        self.focus_score = 50
        
        # Engagement and focus weights by emotion
        self.engagement_weights = {
            'happy': 100,      # Positive engagement
            'surprise': 80,    # Often indicates interest
            'neutral': 60,     # Baseline attention
            'sad': 40,         # Lower engagement
            'angry': 30,       # Might indicate frustration
            'fear': 30,        # Might indicate anxiety
            'disgust': 20      # Typically negative engagement
        }
        
        self.focus_weights = {
            'neutral': 100,    # Strong focus is often neutral-faced
            'surprise': 80,    # Can indicate concentration
            'happy': 70,       # Positive but may be distracted
            'angry': 60,       # Might be focused but negatively
            'sad': 50,         # May indicate lower focus
            'fear': 40,        # Might be distracted by anxiety
            'disgust': 30      # Typically indicates distraction
        }
    
    def analyze_emotion(self, frame):
        """
        Start emotion analysis in background thread if enough time has passed.
        
        Args:
            frame: The video frame to analyze
        """
        current_time = time.time()
        
        # Only run detection at intervals to reduce CPU load
        if (current_time - self.last_detection_time >= self.detection_interval 
                and not self.processing and frame is not None):
            self.processing = True
            
            # Make a copy of the frame to prevent modification
            process_frame = frame.copy()
            
            # Start detection in background thread
            self.emotion_thread = threading.Thread(
                target=self._detect_emotion_thread,
                args=(process_frame,)
            )
            self.emotion_thread.daemon = True
            self.emotion_thread.start()
    
    def _detect_emotion_thread(self, frame):
        """Process emotion detection in background thread"""
        try:
            # Resize to speed up processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                img_path=small_frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Handle different result formats
            if isinstance(result, dict):
                self.emotion = result.get('dominant_emotion', 'neutral')
                self.emotion_scores = result.get('emotion', {})
            elif isinstance(result, list) and len(result) > 0:
                self.emotion = result[0].get('dominant_emotion', 'neutral')
                self.emotion_scores = result[0].get('emotion', {})
            
            # Update engagement and focus scores
            self.engagement_score = self.engagement_weights.get(self.emotion, 50)
            self.focus_score = self.focus_weights.get(self.emotion, 50)
            
            print(f"Detected emotion: {self.emotion}")
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
        finally:
            self.processing = False
            self.last_detection_time = time.time()


class ComprehensiveAnalyzer:
    """Combines posture and emotion analysis for comprehensive feedback"""
    
    def __init__(self):
        """Initialize the comprehensive analyzer"""
        self.posture_analyzer = PostureAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()
        self.eye_contact_analyzer = EyeContactAnalyzer()
        
        # Font and colors
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'excellent': (0, 255, 0),    # Green
            'good': (0, 255, 255),       # Yellow
            'fair': (0, 165, 255),       # Orange
            'poor': (0, 0, 255)          # Red
        }
    
    def get_status_color(self, score):
        """Get color based on score category"""
        if score >= 80:
            return self.colors['excellent'], "Excellent"
        elif score >= 60:
            return self.colors['good'], "Good"
        elif score >= 40:
            return self.colors['fair'], "Fair"
        else:
            return self.colors['poor'], "Poor"
    
    def draw_score_bar(self, frame, y_position, score, label):
        """Draw a visual score bar with label"""
        h, w = frame.shape[:2]
        bar_start_x = 180
        bar_width = 200
        bar_height = 20
        
        # Draw label
        cv2.putText(
            frame,
            label,
            (10, y_position),
            self.font,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Draw empty bar background
        cv2.rectangle(
            frame,
            (bar_start_x, y_position - 15),
            (bar_start_x + bar_width, y_position),
            (80, 80, 80),
            cv2.FILLED
        )
        
        # Calculate filled width and get appropriate color
        filled_width = int(bar_width * score / 100)
        color, status = self.get_status_color(score)
        
        # Draw filled portion of bar
        cv2.rectangle(
            frame,
            (bar_start_x, y_position - 15),
            (bar_start_x + filled_width, y_position),
            color,
            cv2.FILLED
        )
        
        # Draw score text
        cv2.putText(
            frame,
            f"{score}% ({status})",
            (bar_start_x + bar_width + 10, y_position - 2),
            self.font,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
    
    def calculate_composite_score(self, posture_score, engagement_score, focus_score, eye_contact_score):
        """
        Calculate a weighted composite score from all metrics.
        
        Args:
            posture_score: Posture score (0-100)
            engagement_score: Engagement score (0-100)
            focus_score: Focus score (0-100)
            eye_contact_score: Eye contact score (0-100)
            
        Returns:
            Composite score (0-100)
        """
        # Weights can be adjusted based on importance
        weights = {
            'posture': 0.3,
            'engagement': 0.2,
            'focus': 0.2,
            'eye_contact': 0.3  # Eye contact is important for presentations
        }
        
        composite_score = int(
            posture_score * weights['posture'] +
            engagement_score * weights['engagement'] +
            focus_score * weights['focus'] +
            eye_contact_score * weights['eye_contact']
        )
        
        return min(100, max(0, composite_score))
    
    def process_frame(self, frame, draw_annotations=True):
        """
        Process a video frame with both posture and emotion analysis.
        
        Args:
            frame: The video frame to process
            draw_annotations: Whether to draw score bars and text on the frame
            
        Returns:
            tuple: (processed_frame, scores_dict)
        """
        # Copy frame to avoid modifications
        processed_frame = frame.copy()
        
        # Analyze posture
        processed_frame, posture_score = self.posture_analyzer.analyze_posture(processed_frame)
        
        # Analyze eye contact
        processed_frame, eye_contact_score = self.eye_contact_analyzer.analyze_eye_contact(processed_frame)
        
        # Start emotion analysis in background
        self.emotion_analyzer.analyze_emotion(frame)
        
        # Get current scores
        engagement_score = self.emotion_analyzer.engagement_score
        focus_score = self.emotion_analyzer.focus_score
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(
            posture_score, engagement_score, focus_score, eye_contact_score
        )
        
        # Create scores dictionary
        scores = {
            "posture_score": posture_score,
            "engagement_score": engagement_score,
            "focus_score": focus_score,
            "eye_contact_score": eye_contact_score,
            "composite_score": composite_score,
            "emotion": self.emotion_analyzer.emotion if self.emotion_analyzer.emotion else "neutral"
        }
        
        # Only draw annotations if requested
        if draw_annotations:
            # Add emotion info
            if self.emotion_analyzer.emotion:
                cv2.putText(
                    processed_frame,
                    f"Emotion: {self.emotion_analyzer.emotion.capitalize()}",
                    (10, 60),
                    self.font,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Draw score bars for all metrics
            self.draw_score_bar(processed_frame, 100, posture_score, "Posture:")
            self.draw_score_bar(processed_frame, 130, engagement_score, "Engagement:")
            self.draw_score_bar(processed_frame, 160, focus_score, "Focus:")
            self.draw_score_bar(processed_frame, 190, eye_contact_score, "Eye Contact:")
            self.draw_score_bar(processed_frame, 230, composite_score, "OVERALL:")
        
        return processed_frame, scores


def main():
    """Test function for comprehensive analysis without GUI"""
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print(f"Successfully opened camera at index 1")
    else:
        print("Error: Could not open any video source.")
        return
    
    time.sleep(2)
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer()
    
    print("Starting comprehensive analysis...")
    print("Running for 10 frames, then exiting")
    
    # Process a few frames as a test
    for i in range(10):
        # Read frame
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
        
        # Process frame with comprehensive analysis (no GUI drawing)
        _, scores = analyzer.process_frame(frame, draw_annotations=False)
        
        # Print the scores
        print(f"Frame {i+1} scores:")
        for key, value in scores.items():
            print(f"  {key}: {value}")
        
        # Small delay
        time.sleep(0.1)
    
    # Cleanup
    cap.release()
    print("Test complete")


if __name__ == "__main__":
    main()
