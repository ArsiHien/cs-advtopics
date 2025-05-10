from enum import Enum
from typing import Tuple, Union, Optional

import cv2
import numpy as np
import yaml


def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Get monitor dimensions in millimeters and pixels.
    :return: tuple of monitor width and height in mm and pixels or None
    """
    # Method 1: Try using screeninfo if available
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        if monitors:
            # Use the primary monitor if available, otherwise the first one
            primary_monitors = [m for m in monitors if m.is_primary]
            monitor = primary_monitors[0] if primary_monitors else monitors[0]

            w_pixels, h_pixels = monitor.width, monitor.height

            # If monitor provides physical dimensions
            if hasattr(monitor, 'width_mm') and hasattr(monitor, 'height_mm') and monitor.width_mm and monitor.height_mm:
                w_mm, h_mm = monitor.width_mm, monitor.height_mm
            else:
                # Approximate using standard 96 DPI (~ 0.264583 mm per pixel)
                w_mm = int(w_pixels * 0.264583)
                h_mm = int(h_pixels * 0.264583)

            return (w_mm, h_mm), (w_pixels, h_pixels)
    except (ImportError, Exception) as e:
        pass  # Fall back to other methods

    # Method 2: Try Windows-specific method
    try:
        import ctypes
        user32 = ctypes.windll.user32

        # Get display dimensions in pixels
        w_pixels = user32.GetSystemMetrics(0)  # SM_CXSCREEN
        h_pixels = user32.GetSystemMetrics(1)  # SM_CYSCREEN

        # Try to get physical size using Windows API if possible
        # GetDeviceCaps with HORZSIZE and VERTSIZE returns physical width/height in mm
        try:
            gdi32 = ctypes.windll.gdi32
            dc = user32.GetDC(None)
            w_mm = gdi32.GetDeviceCaps(dc, 4)  # HORZSIZE
            h_mm = gdi32.GetDeviceCaps(dc, 6)  # VERTSIZE
            user32.ReleaseDC(None, dc)
        except:
            # Use standard 24-inch monitor dimensions as fallback
            w_mm, h_mm = 531, 298  # Approximate size of 24-inch monitor

        return (w_mm, h_mm), (w_pixels, h_pixels)
    except Exception as e:
        pass  # Fall back to next method

    # Method 3: Try using tkinter as a last resort
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window

        w_pixels = root.winfo_screenwidth()
        h_pixels = root.winfo_screenheight()

        # Convert to mm (approximate)
        w_mm = int(w_pixels * 0.264583)  # 1 inch = 25.4 mm, assume 96 DPI
        h_mm = int(h_pixels * 0.264583)

        root.destroy()
        return (w_mm, h_mm), (w_pixels, h_pixels)
    except Exception as e:
        print(f"Error getting screen dimensions: {e}")
        return None, None


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2


class TargetOrientation(Enum):
    UP = 82
    DOWN = 84
    LEFT = 81
    RIGHT = 83


def get_camera_matrix(calibration_matrix_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera_matrix and dist_coefficients from `calibration_matrix_path`.

    :param calibration_matrix_path: path to calibration file
    :return: camera intrinsic matrix and dist_coefficients
    """
    with open(calibration_matrix_path, 'r') as file:
        calibration_matrix = yaml.safe_load(file)
    camera_matrix = np.asarray(
        calibration_matrix['camera_matrix']).reshape(3, 3)
    dist_coefficients = np.asarray(calibration_matrix['dist_coeff'])
    return camera_matrix, dist_coefficients


def get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, shape, results, face_model, face_model_all, landmarks_ids):
    """
    Fit `face_model` onto `face_landmarks` using `solvePnP`.

    :param camera_matrix: camera intrinsic matrix
    :param dist_coefficients: distortion coefficients
    :param shape: image shape
    :param results: output of MediaPipe FaceMesh
    :return: full face model in the camera coordinate system
    """
    height, width, _ = shape
    face_landmarks = np.asarray([[landmark.x * width, landmark.y * height]
                                for landmark in results.multi_face_landmarks[0].landmark])
    face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

    rvec, tvec = None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients,
                                                      rvec=rvec, tvec=tvec, useExtrinsicGuess=rvec is not None, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
    for _ in range(10):
        success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec,
                                           # Second fit for higher accuracy
                                           tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))
    # 3D positions of facial landmarks
    return np.dot(head_rotation_matrix, face_model.T) + tvec.reshape((3, 1)), np.dot(head_rotation_matrix, face_model_all.T) + tvec.reshape((3, 1))


def gaze_2d_to_3d(gaze: np.ndarray) -> np.ndarray:
    """
    pitch and gaze to 3d vector

    :param gaze: pitch and gaze vector
    :return: 3d vector
    """
    x = -np.cos(gaze[0]) * np.sin(gaze[1])
    y = -np.sin(gaze[0])
    z = -np.cos(gaze[0]) * np.cos(gaze[1])
    return np.array([x, y, z])


def ray_plane_intersection(support_vector: np.ndarray, direction_vector: np.ndarray, plane_normal: np.ndarray, plane_d: np.ndarray) -> np.ndarray:
    """
    Calulate the intersection between the gaze ray and the plane that represents the monitor.

    :param support_vector: support vector of the gaze
    :param direction_vector: direction vector of the gaze
    :param plane_normal: normal of the plane
    :param plane_d: d of the plane
    :return: point in 3D where the the person is looking at on the screen
    """
    # Handle zero cases in direction vector to avoid division by zero
    eps = 1e-10
    direction_vector = np.array([
        max(abs(direction_vector[0]), eps) *
        np.sign(direction_vector[0] + eps),
        max(abs(direction_vector[1]), eps) *
        np.sign(direction_vector[1] + eps),
        max(abs(direction_vector[2]), eps) * np.sign(direction_vector[2] + eps)
    ])

    a11 = direction_vector[1]
    a12 = -direction_vector[0]
    b1 = direction_vector[1] * support_vector[0] - \
        direction_vector[0] * support_vector[1]

    a22 = direction_vector[2]
    a23 = -direction_vector[1]
    b2 = direction_vector[2] * support_vector[1] - \
        direction_vector[1] * support_vector[2]

    line_w = np.array([[a11, a12, 0], [0, a22, a23]])
    line_b = np.array([[b1], [b2]])

    matrix = np.insert(line_w, 2, plane_normal, axis=0)
    bias = np.insert(line_b, 2, plane_d, axis=0)

    try:
        return np.linalg.solve(matrix, bias).reshape(3)
    except np.linalg.LinAlgError:
        # Fallback if singular matrix
        print("Warning: Singular matrix in ray-plane intersection, using fallback")
        return np.array([0, 0, 0])


def plane_equation(rmat: np.ndarray, tmat: np.ndarray) -> np.ndarray:
    """
    Computes the equation of x-y plane.
    The normal vector of the plane is z-axis in rotation matrix. And tmat provide on point in the plane.

    :param rmat: rotation matrix
    :param tmat: translation matrix
    :return: (a, b, c, d), where the equation of plane is ax + by + cz = d
    """

    assert type(rmat) == type(np.zeros(0)) and rmat.shape == (
        3, 3), "There is an error about rmat."
    assert type(tmat) == type(
        np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

    n = rmat[:, 2]
    origin = np.reshape(tmat, (3))

    a = n[0]
    b = n[1]
    c = n[2]

    d = origin[0] * n[0] + origin[1] * n[1] + origin[2] * n[2]
    return np.array([a, b, c, d])


# def get_point_on_screen(monitor_mm: Tuple[float, float], monitor_pixels: Tuple[float, float], result: np.ndarray) -> Tuple[int, int]:
#     """
#     Calculate point in screen in pixels.

#     :param monitor_mm: dimensions of the monitor in mm
#     :param monitor_pixels: dimensions of the monitor in pixels
#     :param result: predicted point on the screen in mm
#     :return: point in screen in pixels
#     """
#     result_x = result[0]
#     result_x = -result_x + monitor_mm[0] / 2
#     result_x = result_x * (monitor_pixels[0] / monitor_mm[0])
#     # Clamp to screen bounds
#     result_x = max(0, min(result_x, monitor_pixels[0]))

#     result_y = result[1]
#     result_y = result_y - 20  # 20 mm offset
#     result_y = min(result_y, monitor_mm[1])
#     result_y = result_y * (monitor_pixels[1] / monitor_mm[1])
#     # Clamp to screen bounds
#     result_y = max(0, min(result_y, monitor_pixels[1]))

#     return tuple(np.asarray([result_x, result_y]).round().astype(int))

def get_point_on_screen(monitor_mm: Tuple[float, float], monitor_pixels: Tuple[float, float], result: np.ndarray) -> Tuple[int, int]:
    result_x = result[0]
    result_x = -result_x + monitor_mm[0] / 2

    # Tính tỷ lệ với chiều rộng màn hình
    ratio_x = result_x / (monitor_mm[0] / 2)

    # Điều chỉnh theo tỷ lệ đó để làm mượt chuyển động
    scaled_x = ratio_x * (monitor_pixels[0] / 2)
    scaled_x = max(0, min(scaled_x, monitor_pixels[0]))  # Clamp to screen bounds

    result_y = result[1]
    result_y = result_y - 20  # 20 mm offset
    result_y = min(result_y, monitor_mm[1])

    # Sử dụng toàn bộ chiều cao màn hình để tính tỷ lệ
    ratio_y = (result_y - monitor_mm[1] / 2) / (monitor_mm[1] / 2)

    # Điều chỉnh Y theo tỷ lệ
    scaled_y = (ratio_y + 1) * (monitor_pixels[1] / 2)
    scaled_y = max(0, min(scaled_y, monitor_pixels[1])) * 2  # Clamp to screen bounds

    return tuple(np.asarray([scaled_x, scaled_y]).round().astype(int))