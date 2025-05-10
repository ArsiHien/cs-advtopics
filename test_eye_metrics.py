import cv2
from eye_metrics import EyeMetricsDetector
import time


def has_gui_support():
    """Check if OpenCV has GUI support (not headless version)."""
    try:
        # Try to create a test window - will fail on headless OpenCV
        test_window_name = "__test_window__"
        cv2.namedWindow(test_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow(test_window_name)
        return True
    except cv2.error:
        print("OpenCV GUI support not available. Running in headless mode.")
        return False


def main():
    """
    Standalone test for the EyeMetricsDetector functionality
    """
    # Check for GUI support
    HAS_GUI = has_gui_support()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize the eye metrics detector
    detector = EyeMetricsDetector()

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for eye metrics
        processed_frame, distance, blink_count, is_blinking = detector.process_frame(
            frame)

        # Display the processed frame if GUI is available
        if HAS_GUI:
            cv2.imshow('Eye Metrics Test', processed_frame)

        # Print metrics
        status = "Blinking" if is_blinking else "Open"
        distance_str = f"{distance:.2f} cm" if distance else "Unknown"
        print(
            f"Eye Status: {status}, Distance: {distance_str}, Blinks: {blink_count}")

        # Exit if ESC key is pressed or after 100 iterations in headless mode
        if HAS_GUI:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            # In headless mode, just wait a bit
            time.sleep(0.1)

    # Release resources
    cap.release()
    if HAS_GUI:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
