import collections
import time

import cv2
import numpy as np

# Import the EyeMetricsDetector
from eye_metrics import EyeMetricsDetector

# Check if OpenCV has GUI support (import from main.py if available)
try:
    from main import HAS_GUI_SUPPORT, safe_destroyAllWindows
except ImportError:
    # Fallback definition if not imported from main.py
    def has_gui_support():
        """Check if OpenCV has GUI support (not headless version)."""
        try:
            test_window_name = "__test_window__"
            cv2.namedWindow(test_window_name, cv2.WINDOW_AUTOSIZE)
            cv2.destroyWindow(test_window_name)
            return True
        except cv2.error:
            print(
                "OpenCV GUI support not available in WebcamSource. Running in headless mode.")
            return False

    HAS_GUI_SUPPORT = has_gui_support()

    def safe_destroyAllWindows():
        """Destroy all windows only if GUI support is available."""
        if HAS_GUI_SUPPORT:
            return cv2.destroyAllWindows()
        else:
            return None


class WebcamSource:
    """
    Helper class for OpenCV VideoCapture. Can be used as an iterator.
    """

    def __init__(self, camera_id=0, width=1280, height=720, fps=30, buffer_size=1, enable_eye_tracking=True):
        self.__name = "WebcamSource"
        self.__capture = cv2.VideoCapture(camera_id)
        self.__capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.__capture.set(cv2.CAP_PROP_FOURCC,
                           cv2.VideoWriter_fourcc(*'MJPG'))
        self.__capture.set(cv2.CAP_PROP_FPS, fps)
        self.__capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.buffer_size = buffer_size

        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.fps_deque = collections.deque(maxlen=fps)

        # Initialize eye metrics detector if enabled
        self.enable_eye_tracking = enable_eye_tracking
        if self.enable_eye_tracking:
            self.eye_metrics_detector = EyeMetricsDetector()
            self.eye_distance = None
            self.blink_count = 0
            self.is_blinking = False

    def __iter__(self):
        if not self.__capture.isOpened():
            raise StopIteration
        return self

    def __next__(self):
        """
        Get next frame from webcam or stop iteration when no frame can be grabbed from webcam

        :return: None
        """
        ret, frame = self.__capture.read()

        if not ret:
            raise StopIteration

        # Process frame with MediaPipe if eye tracking is enabled
        if self.enable_eye_tracking:
            processed_frame, self.eye_distance, self.blink_count, self.is_blinking = self.eye_metrics_detector.process_frame(
                frame)
            frame = processed_frame

        # Use safe waitKey if GUI support is available
        if HAS_GUI_SUPPORT:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise StopIteration
        else:
            # Small delay to prevent CPU thrashing in headless mode
            time.sleep(0.001)

        return frame

    def clear_frame_buffer(self):
        for _ in range(self.buffer_size):
            self.__capture.read()

    def __del__(self):
        self.__capture.release()
        # Use safe version of destroyAllWindows
        safe_destroyAllWindows()

    def show(self, frame, only_print=False):
        self.new_frame_time = time.time()
        self.fps_deque.append(1 / (self.new_frame_time - self.prev_frame_time))
        self.prev_frame_time = self.new_frame_time

        if only_print:
            print(f'{self.__name} - FPS: {np.mean(self.fps_deque):5.2f}')
        elif HAS_GUI_SUPPORT:  # Only show if GUI support is available
            cv2.imshow('show_frame', frame)
            cv2.setWindowTitle(
                "show_frame", f'{self.__name} - FPS: {np.mean(self.fps_deque):5.2f}')

    def get_eye_metrics(self):
        """Get the current eye metrics from the detector

        Returns:
            tuple: (distance, blink_count, is_blinking)
        """
        if not self.enable_eye_tracking:
            return None, 0, False

        return self.eye_distance, self.blink_count, self.is_blinking

    def reset_blink_counter(self):
        """Reset the blink counter"""
        if self.enable_eye_tracking:
            self.eye_metrics_detector.reset_blink_counter()
            self.blink_count = 0
