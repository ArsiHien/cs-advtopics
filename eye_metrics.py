import cv2
import mediapipe as mp
import numpy as np
import time


class EyeMetricsDetector:
    """
    Class for detecting eye metrics such as:
    - Eye distance from screen
    - Blink detection and counting
    using MediaPipe Face Mesh
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Define eye landmarks indices
        # Left eye landmarks
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

        # For blink detection
        self.blink_counter = 0
        self.blink_detected = False
        self.last_blink_time = time.time()
        self.EYE_AR_THRESHOLD = 0.2  # Eye aspect ratio threshold for blink detection
        self.EYE_AR_CONSEC_FRAMES = 2  # Number of consecutive frames for blink
        self.blink_counter_frames = 0

        # For distance estimation
        self.KNOWN_DISTANCE = 50.0  # cm
        self.KNOWN_WIDTH_BETWEEN_EYES = 6.3  # cm - average human interocular distance

        # For smoothing
        self.distance_history = []
        self.max_history = 10

    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate the eye aspect ratio (EAR) which is used for blink detection
        """
        # Calculate vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

        # Calculate horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _estimate_distance_from_screen(self, image, face_landmarks):
        """
        Estimate the distance from the face to the screen using interocular distance
        """
        if face_landmarks:
            # Get face dimensions in the image
            h, w, _ = image.shape

            # Get the positions of the eyes
            left_eye = np.array([int(face_landmarks.landmark[386].x * w),
                                 int(face_landmarks.landmark[386].y * h)])
            right_eye = np.array([int(face_landmarks.landmark[159].x * w),
                                 int(face_landmarks.landmark[159].y * h)])

            # Calculate the distance between eyes in pixels
            pixel_eye_distance = np.linalg.norm(left_eye - right_eye)

            # Avoid division by zero
            if pixel_eye_distance == 0:
                return None

            # Calculate the distance in cm using similar triangles formula
            # Distance = (Known Width * Focal Length) / Pixel Width
            distance = (self.KNOWN_WIDTH_BETWEEN_EYES * w) / pixel_eye_distance

            # Add to history for smoothing
            self.distance_history.append(distance)
            if len(self.distance_history) > self.max_history:
                self.distance_history.pop(0)

            # Return the smoothed distance
            return np.mean(self.distance_history)
        return None

    def process_frame(self, frame):
        """
        Process a frame to detect eye metrics
        Returns: 
            - processed_frame: frame with visualizations
            - distance: estimated distance in cm
            - blink_count: total number of blinks detected
            - is_blinking: whether the person is currently blinking
        """
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Process the image
        results = self.face_mesh.process(image_rgb)

        # Create a copy for visualization
        processed_frame = frame.copy()

        # Default return values
        distance = None
        is_blinking = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Draw the face mesh
            self.mp_drawing.draw_landmarks(
                image=processed_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Draw the eye contours
            self.mp_drawing.draw_landmarks(
                image=processed_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Get eye landmarks
            left_eye_landmarks = np.array([
                [face_landmarks.landmark[i].x * w,
                    face_landmarks.landmark[i].y * h]
                for i in self.LEFT_EYE_INDICES
            ])

            right_eye_landmarks = np.array([
                [face_landmarks.landmark[i].x * w,
                    face_landmarks.landmark[i].y * h]
                for i in self.RIGHT_EYE_INDICES
            ])

            # Calculate EAR for both eyes
            left_ear = self._calculate_eye_aspect_ratio(left_eye_landmarks)
            right_ear = self._calculate_eye_aspect_ratio(right_eye_landmarks)

            # Average EAR for both eyes
            ear = (left_ear + right_ear) / 2.0

            # Check for blink
            if ear < self.EYE_AR_THRESHOLD:
                self.blink_counter_frames += 1
                is_blinking = True
            else:
                # If eyes were closed for a sufficient number of frames, count as a blink
                if self.blink_counter_frames >= self.EYE_AR_CONSEC_FRAMES:
                    current_time = time.time()
                    # Prevent counting multiple blinks too quickly (debounce)
                    if current_time - self.last_blink_time > 0.2:  # 200ms debounce
                        self.blink_counter += 1
                        self.last_blink_time = current_time

                self.blink_counter_frames = 0
                is_blinking = False

            # Estimate distance from screen
            distance = self._estimate_distance_from_screen(
                frame, face_landmarks)

            # Draw eye status text
            eye_status = "Blinking" if is_blinking else "Open"
            cv2.putText(processed_frame, f"Eyes: {eye_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw blink counter
            cv2.putText(processed_frame, f"Blinks: {self.blink_counter}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw distance if available
            if distance:
                cv2.putText(processed_frame, f"Distance: {distance:.2f} cm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return processed_frame, distance, self.blink_counter, is_blinking

    def reset_blink_counter(self):
        """Reset the blink counter"""
        self.blink_counter = 0
