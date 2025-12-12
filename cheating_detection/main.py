
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import json
from datetime import datetime
from collections import deque
import pickle
import os

# Scikit-Learn imports for ML model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Matplotlib for visualization
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# YOLO for phone detection
try:
    from ultralytics import YOLO
    yolo_enabled = True
except:
    yolo_enabled = False
    print("YOLO not available. Install: pip install ultralytics")

mp_face_mesh = mp.solutions.face_mesh

# ==================== CONFIGURATION ====================

# 3D MODEL POINTS (STANDARD HEAD MODEL)
model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye outer corner
    (225.0, 170.0, -135.0),      # Right eye outer corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# FaceMesh landmark indices
FACE_IDXS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291
}

# Eye gaze detection indices
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

# Mouth detection
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# Detection thresholds
THRESHOLDS = {
    "look_away_angle": 25,
    "look_down_angle": 20,
    "no_face_time": 3,
    "eye_gaze": 0.3,
    "mouth_open": 0.03,
    "phone_confidence": 0.5,
    "look_away_warning": 3,
    "look_away_critical": 7,
    "look_away_severe": 15
}

# Score weights
SCORE_WEIGHTS = {
    "multiple_faces": 15,
    "no_face": 20,
    "looking_away": 10,
    "looking_down": 8,
    "head_tilt": 5,
    "eye_gaze_away": 12,
    "mouth_open": 7,
    "phone_detected": 25,
    "look_away_warning": 5,
    "look_away_critical": 15,
    "look_away_severe": 30
}


class KalmanFilter:
    """Kalman filter for smoothing angle measurements"""
    def __init__(self, process_variance=1e-3, measurement_variance=5e-1):
        self.x = np.zeros(2)  # State: [angle, velocity]
        self.P = np.eye(2)     # Covariance
        self.Q = np.array([[process_variance, 0],
                          [0, process_variance]])  # Process noise
        self.R = np.array([[measurement_variance]])  # Measurement noise
        self.F = np.array([[1, 1], [0, 1]])  # State transition
        self.H = np.array([[1, 0]])  # Measurement matrix
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        y = measurement - (self.H @ self.x)[0]
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0]
    
    def process(self, measurement):
        self.predict()
        return self.update(measurement)

# ==================== ATTENTION SCORE MODEL ====================

class AttentionScoreModel:
    """ML-based attention scoring using Random Forest"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_history = deque(maxlen=1000)
        
    def extract_features(self, yaw, pitch, roll, eye_gaze, mouth_ratio, 
                        no_face, multiple_faces, phone_detected):
        """Extract features for ML model"""
        return np.array([
            abs(yaw),
            abs(pitch),
            abs(roll),
            abs(eye_gaze - 0.5),  # Distance from center
            mouth_ratio,
            1 if no_face else 0,
            1 if multiple_faces else 0,
            1 if phone_detected else 0
        ])
    
    def train_initial_model(self):
        """Train model with synthetic data (for demonstration)"""
        # Generate training data
        np.random.seed(42)
        n_samples = 500
        
        # Focused behavior (label 1)
        focused_data = np.column_stack([
            np.random.normal(5, 5, n_samples),    # yaw
            np.random.normal(5, 5, n_samples),    # pitch
            np.random.normal(3, 3, n_samples),    # roll
            np.random.normal(0.1, 0.05, n_samples),  # eye_gaze_diff
            np.random.normal(0.01, 0.005, n_samples),  # mouth
            np.zeros(n_samples),  # no_face
            np.zeros(n_samples),  # multiple_faces
            np.zeros(n_samples)   # phone
        ])
        
        # Distracted behavior (label 0)
        distracted_data = np.column_stack([
            np.random.normal(35, 15, n_samples),
            np.random.normal(25, 10, n_samples),
            np.random.normal(15, 8, n_samples),
            np.random.normal(0.3, 0.1, n_samples),
            np.random.normal(0.04, 0.02, n_samples),
            np.random.binomial(1, 0.2, n_samples),
            np.random.binomial(1, 0.1, n_samples),
            np.random.binomial(1, 0.15, n_samples)
        ])
        
        X = np.vstack([focused_data, distracted_data])
        y = np.hstack([np.ones(n_samples), np.zeros(n_samples)])
        
        # Train model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("✓ Attention Score Model trained successfully")
    
    def predict_attention(self, features):
        """Predict attention score (0-100)"""
        if not self.is_trained:
            self.train_initial_model()
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.model.predict_proba(features_scaled)[0]
        attention_score = proba[1] * 100  # Probability of being focused
        return attention_score
    
    def save_model(self, filepath="attention_model.pkl"):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load_model(self, filepath="attention_model.pkl"):
        """Load trained model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True

# ==================== DATA LOGGER & VISUALIZATION ====================

class SessionAnalyzer:
    """Analyze and visualize session data"""
    def __init__(self):
        self.timestamps = []
        self.attention_scores = []
        self.yaw_angles = []
        self.pitch_angles = []
        self.cheating_scores = []
        self.events = []
        
    def add_data(self, timestamp, attention_score, yaw, pitch, cheating_score):
        self.timestamps.append(timestamp)
        self.attention_scores.append(attention_score)
        self.yaw_angles.append(yaw)
        self.pitch_angles.append(pitch)
        self.cheating_scores.append(cheating_score)
    
    def add_event(self, timestamp, event_type):
        self.events.append((timestamp, event_type))
    
    def generate_report(self, output_dir="session_reports"):
        """Generate comprehensive analysis report with graphs"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cheating Detection Session Analysis', fontsize=16, fontweight='bold')
        
        # Convert timestamps to relative seconds
        if len(self.timestamps) > 0:
            start_time = self.timestamps[0]
            time_seconds = [(t - start_time) for t in self.timestamps]
            
            # Plot 1: Attention Score over Time
            axes[0, 0].plot(time_seconds, self.attention_scores, 'g-', linewidth=2)
            axes[0, 0].axhline(y=50, color='r', linestyle='--', label='Threshold')
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Attention Score (%)')
            axes[0, 0].set_title('Attention Score Timeline')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Plot 2: Head Pose Angles
            axes[0, 1].plot(time_seconds, self.yaw_angles, 'b-', label='Yaw', alpha=0.7)
            axes[0, 1].plot(time_seconds, self.pitch_angles, 'r-', label='Pitch', alpha=0.7)
            axes[0, 1].axhline(y=THRESHOLDS["look_away_angle"], color='orange', linestyle='--', alpha=0.5)
            axes[0, 1].axhline(y=-THRESHOLDS["look_away_angle"], color='orange', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Angle (degrees)')
            axes[0, 1].set_title('Head Pose Angles')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Cheating Score over Time
            axes[1, 0].plot(time_seconds, self.cheating_scores, 'r-', linewidth=2)
            axes[1, 0].fill_between(time_seconds, self.cheating_scores, alpha=0.3, color='red')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Cheating Score')
            axes[1, 0].set_title('Cheating Score Timeline')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Event Distribution
            event_types = {}
            for _, event_type in self.events:
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            if event_types:
                events_list = list(event_types.keys())
                counts = list(event_types.values())
                axes[1, 1].barh(events_list, counts, color='coral')
                axes[1, 1].set_xlabel('Count')
                axes[1, 1].set_title('Cheating Events Distribution')
                axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        report_path = os.path.join(output_dir, f'session_report_{timestamp_str}.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Session report saved: {report_path}")
        
        # Generate text summary
        self.generate_text_summary(output_dir, timestamp_str)
        
        return report_path
    
    def generate_text_summary(self, output_dir, timestamp_str):
        """Generate text summary of session"""
        summary_path = os.path.join(output_dir, f'session_summary_{timestamp_str}.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CHEATING DETECTION SESSION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            if len(self.attention_scores) > 0:
                f.write(f"Session Duration: {len(self.timestamps)} frames\n")
                f.write(f"Average Attention Score: {np.mean(self.attention_scores):.2f}%\n")
                f.write(f"Minimum Attention Score: {np.min(self.attention_scores):.2f}%\n")
                f.write(f"Average Cheating Score: {np.mean(self.cheating_scores):.2f}\n")
                f.write(f"Maximum Cheating Score: {np.max(self.cheating_scores):.2f}\n\n")
                
                f.write("Event Summary:\n")
                event_types = {}
                for _, event_type in self.events:
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                for event, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {event}: {count} times\n")
        
        print(f"✓ Text summary saved: {summary_path}")

# ==================== MAIN DETECTION SYSTEM ====================

class CheatingDetectionSystem:
    """Complete cheating detection system"""
    def __init__(self):
        # Initialize Kalman filters
        self.kalman_yaw = KalmanFilter()
        self.kalman_pitch = KalmanFilter()
        self.kalman_roll = KalmanFilter()
        
        # Initialize ML model
        self.attention_model = AttentionScoreModel()
        
        # Initialize analyzer
        self.analyzer = SessionAnalyzer()
        
        # State tracking
        self.looking_away_state = {
            "is_looking_away": False,
            "start_time": None,
            "duration": 0,
            "total_time": 0,
            "warning_triggered": False,
            "critical_triggered": False,
            "severe_triggered": False
        }
        
        self.last_face_time = time.time()
        self.cheating_score = 0
        self.score_decay_rate = 0.5
        self.last_score_update = time.time()
        
        # CSV logging
        self.csv_filename = f"cheating_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Event', 'Score', 'Attention_Score', 
                           'Yaw', 'Pitch', 'Roll', 'Details'])
        
        # Load YOLO if available
        if yolo_enabled:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("✓ YOLO model loaded")
            except:
                self.yolo_model = None
                print("⚠ YOLO model not loaded")
        else:
            self.yolo_model = None
    
    def log_event(self, event, score, attention_score, yaw, pitch, roll, details=""):
        """Log event to CSV"""
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                event, score, f"{attention_score:.1f}%",
                f"{yaw:.2f}", f"{pitch:.2f}", f"{roll:.2f}", details
            ])
    
    def get_eye_gaze_ratio(self, landmarks, eye_indices, outer_idx, inner_idx, w, h):
        """Calculate eye gaze direction ratio"""
        iris_x = np.mean([landmarks[i].x for i in eye_indices]) * w
        outer_x = landmarks[outer_idx].x * w
        inner_x = landmarks[inner_idx].x * w
        
        eye_width = abs(inner_x - outer_x)
        if eye_width == 0:
            return 0.5
        
        ratio = (iris_x - outer_x) / eye_width
        return ratio
    
    def get_mouth_aspect_ratio(self, landmarks, w, h):
        """Calculate mouth opening ratio"""
        top = landmarks[MOUTH_TOP]
        bottom = landmarks[MOUTH_BOTTOM]
        mouth_height = abs((bottom.y - top.y) * h)
        face_height = h * 0.8
        return mouth_height / face_height
    
    def update_looking_away_state(self, is_looking_away, current_time):
        """Track looking away duration"""
        alerts = []
        
        if is_looking_away:
            if not self.looking_away_state["is_looking_away"]:
                self.looking_away_state["is_looking_away"] = True
                self.looking_away_state["start_time"] = current_time
                self.looking_away_state["warning_triggered"] = False
                self.looking_away_state["critical_triggered"] = False
                self.looking_away_state["severe_triggered"] = False
            
            self.looking_away_state["duration"] = current_time - self.looking_away_state["start_time"]
            
            if (self.looking_away_state["duration"] >= THRESHOLDS["look_away_severe"] 
                and not self.looking_away_state["severe_triggered"]):
                alerts.append("SEVERE: Looking Away > 15s")
                self.cheating_score += SCORE_WEIGHTS["look_away_severe"]
                self.looking_away_state["severe_triggered"] = True
                
            elif (self.looking_away_state["duration"] >= THRESHOLDS["look_away_critical"] 
                  and not self.looking_away_state["critical_triggered"]):
                alerts.append("CRITICAL: Looking Away > 7s")
                self.cheating_score += SCORE_WEIGHTS["look_away_critical"]
                self.looking_away_state["critical_triggered"] = True
                
            elif (self.looking_away_state["duration"] >= THRESHOLDS["look_away_warning"] 
                  and not self.looking_away_state["warning_triggered"]):
                alerts.append("WARNING: Looking Away > 3s")
                self.cheating_score += SCORE_WEIGHTS["look_away_warning"]
                self.looking_away_state["warning_triggered"] = True
        else:
            if self.looking_away_state["is_looking_away"]:
                self.looking_away_state["total_time"] += self.looking_away_state["duration"]
                self.looking_away_state["is_looking_away"] = False
                self.looking_away_state["duration"] = 0
        
        return alerts
    
    def process_frame(self, frame, face_mesh):
        """Process single frame"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        current_time = time.time()
        alerts = []
        is_looking_away = False
        
        # Decay score
        time_diff = current_time - self.last_score_update
        self.cheating_score = max(0, self.cheating_score - (self.score_decay_rate * time_diff))
        self.last_score_update = current_time
        
        # Initialize default values
        yaw, pitch, roll = 0, 0, 0
        eye_gaze = 0.5
        mouth_ratio = 0
        no_face = False
        multiple_faces = False
        phone_detected = False
        attention_score = 100
        
        # Phone detection
        if self.yolo_model is not None:
            yolo_results = self.yolo_model(frame, verbose=False)
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 67 and conf > THRESHOLDS["phone_confidence"]:
                        phone_detected = True
                        is_looking_away = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"PHONE {conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        alerts.append("Phone Detected!")
                        self.cheating_score += SCORE_WEIGHTS["phone_detected"]
        
        # Face detection
        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) > 1:
                multiple_faces = True
                alerts.append("Multiple Faces!")
                self.cheating_score += SCORE_WEIGHTS["multiple_faces"]
            else:
                self.last_face_time = current_time
                face = results.multi_face_landmarks[0]
                lm = face.landmark
                
                # Get 2D image points
                image_points = np.array([
                    (lm[FACE_IDXS["nose"]].x * w, lm[FACE_IDXS["nose"]].y * h),
                    (lm[FACE_IDXS["chin"]].x * w, lm[FACE_IDXS["chin"]].y * h),
                    (lm[FACE_IDXS["left_eye"]].x * w, lm[FACE_IDXS["left_eye"]].y * h),
                    (lm[FACE_IDXS["right_eye"]].x * w, lm[FACE_IDXS["right_eye"]].y * h),
                    (lm[FACE_IDXS["left_mouth"]].x * w, lm[FACE_IDXS["left_mouth"]].y * h),
                    (lm[FACE_IDXS["right_mouth"]].x * w, lm[FACE_IDXS["right_mouth"]].y * h)
                ], dtype="double")
                
                # Camera matrix
                focal_length = w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ], dtype="double")
                
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(
                    model_points, image_points, cam_matrix,
                    distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                # Get Euler angles
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                proj_matrix = np.hstack((rot_mat, trans_vec))
                _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)
                pitch_raw, yaw_raw, roll_raw = euler.flatten()
                
                # Apply Kalman filter
                yaw = self.kalman_yaw.process(yaw_raw)
                pitch = self.kalman_pitch.process(pitch_raw)
                roll = self.kalman_roll.process(roll_raw)
                
                # Eye gaze detection
                left_gaze = self.get_eye_gaze_ratio(lm, LEFT_EYE_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, w, h)
                right_gaze = self.get_eye_gaze_ratio(lm, RIGHT_EYE_IRIS, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, w, h)
                eye_gaze = (left_gaze + right_gaze) / 2
                
                if eye_gaze < (0.5 - THRESHOLDS["eye_gaze"]):
                    alerts.append("Eyes Looking Left")
                    self.cheating_score += SCORE_WEIGHTS["eye_gaze_away"]
                elif eye_gaze > (0.5 + THRESHOLDS["eye_gaze"]):
                    alerts.append("Eyes Looking Right")
                    self.cheating_score += SCORE_WEIGHTS["eye_gaze_away"]
                
                # Mouth detection
                mouth_ratio = self.get_mouth_aspect_ratio(lm, w, h)
                if mouth_ratio > THRESHOLDS["mouth_open"]:
                    alerts.append("Mouth Open")
                    self.cheating_score += SCORE_WEIGHTS["mouth_open"]
                
                # Head pose detection
                if yaw > THRESHOLDS["look_away_angle"]:
                    alerts.append("Looking Right")
                    is_looking_away = True
                    self.cheating_score += SCORE_WEIGHTS["looking_away"]
                elif yaw < -THRESHOLDS["look_away_angle"]:
                    alerts.append("Looking Left")
                    is_looking_away = True
                    self.cheating_score += SCORE_WEIGHTS["looking_away"]
                
                if pitch > THRESHOLDS["look_down_angle"]:
                    alerts.append("Looking Down")
                    is_looking_away = True
                    self.cheating_score += SCORE_WEIGHTS["looking_down"]
                
                if abs(roll) > 25:
                    alerts.append("Head Tilted")
                    self.cheating_score += SCORE_WEIGHTS["head_tilt"]
                
                # Display angles
                cv2.putText(frame, f"Yaw:   {yaw:.1f}°", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f}°", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Roll:  {roll:.1f}°", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            # No face detected
            if current_time - self.last_face_time > THRESHOLDS["no_face_time"]:
                no_face = True
                is_looking_away = True
                alerts.append("NO FACE DETECTED!")
                self.cheating_score += SCORE_WEIGHTS["no_face"]
        
        # Update looking away state
        looking_away_alerts = self.update_looking_away_state(is_looking_away, current_time)
        alerts.extend(looking_away_alerts)
        
        # Get ML-based attention score
        features = self.attention_model.extract_features(
            yaw, pitch, roll, eye_gaze, mouth_ratio,
            no_face, multiple_faces, phone_detected
        )
        attention_score = self.attention_model.predict_attention(features)
        
        # Log data for analysis
        self.analyzer.add_data(current_time, attention_score, yaw, pitch, self.cheating_score)
        for alert in alerts:
            self.analyzer.add_event(current_time, alert)
        
        # Cap cheating score
        self.cheating_score = min(self.cheating_score, 100)
        
        # Display UI
        self.draw_ui(frame, attention_score, alerts, w, h)
        
        return frame
    
    def draw_ui(self, frame, attention_score, alerts, w, h):
        """Draw user interface"""
        # Attention score
        score_color = (0, 255, 0) if attention_score > 70 else (0, 165, 255) if attention_score > 40 else (0, 0, 255)
        cv2.putText(frame, f"Attention: {attention_score:.1f}%", (w - 300, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
        
        # Looking away duration
        duration_y = 165
        cv2.putText(frame, "=== LOOKING AWAY ===", (20, duration_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        duration_y += 35
        
        if self.looking_away_state["is_looking_away"]:
            duration = self.looking_away_state["duration"]
            dur_color = (0, 255, 0)
            if duration >= THRESHOLDS["look_away_severe"]:
                dur_color = (0, 0, 255)
            elif duration >= THRESHOLDS["look_away_critical"]:
                dur_color = (0, 100, 255)
            elif duration >= THRESHOLDS["look_away_warning"]:
                dur_color = (0, 165, 255)
            
            cv2.putText(frame, f"Duration: {duration:.1f}s", (20, duration_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, dur_color, 2)
        else:
            cv2.putText(frame, "Duration: 0.0s", (20, duration_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Cheating score bar
        if self.cheating_score < 30:
            score_color = (0, 255, 0)
            status = "CLEAR"
        elif self.cheating_score < 60:
            score_color = (0, 165, 255)
            status = "SUSPICIOUS"
        else:
            score_color = (0, 0, 255)
            status = "CHEATING"
        
        bar_width = int((self.cheating_score / 100) * 400)
        cv2.rectangle(frame, (20, h - 80), (420, h - 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 80), (20 + bar_width, h - 40), score_color, -1)
        cv2.rectangle(frame, (20, h - 80), (420, h - 40), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Cheating Score: {int(self.cheating_score)} - {status}",
                   (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
        
        # Alerts
        y_offset = h - 150
        for alert in alerts[:5]:
            cv2.putText(frame, alert, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset -= 35
        
        # Instructions
        cv2.putText(frame, "ESC: Exit | R: Generate Report | F: Fullscreen",
                   (w - 550, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Run the detection system"""
        cap = cv2.VideoCapture(0)
        
        cv2.namedWindow("Cheating Detection System", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Cheating Detection System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("\n" + "="*60)
        print("ONLINE CHEATING DETECTION SYSTEM - STARTED")
        print("="*60)
        print("Controls:")
        print("  ESC  - Exit and generate report")
        print("  R    - Generate report immediately")
        print("  F    - Toggle fullscreen")
        print("="*60 + "\n")
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as face_mesh:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = self.process_frame(frame, face_mesh)
                
                cv2.imshow("Cheating Detection System", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r') or key == ord('R'):  # Generate report
                    print("\n⏳ Generating report...")
                    self.analyzer.generate_report()
                elif key == ord('f') or key == ord('F'):  # Toggle fullscreen
                    cv2.setWindowProperty("Cheating Detection System",
                                        cv2.WND_PROP_FULLSCREEN,
                                        cv2.WINDOW_NORMAL)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        print("\n" + "="*60)
        print("SESSION ENDED - Generating Final Report")
        print("="*60)
        self.analyzer.generate_report()
        print(f"\n✓ CSV log saved: {self.csv_filename}")
        print(f"✓ Total looking away time: {self.looking_away_state['total_time']:.1f}s")
        print(f"✓ Final cheating score: {int(self.cheating_score)}")
        print("="*60 + "\n")

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    system = CheatingDetectionSystem()
    system.run()