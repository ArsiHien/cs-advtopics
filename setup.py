from setuptools import setup, find_packages

setup(
    name="gaze_tracker",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "opencv-python",
        "numpy",
        "mediapipe",
        "albumentations",
        "PyQt5",
        "matplotlib",
        "screeninfo",
    ],
    entry_points={
        "console_scripts": [
            "gaze-tracker=src.core.app:start_application",
            "gaze-launcher=src.ui.launcher:launch_application",
        ],
    },
    description="A gaze tracking pipeline to detect where a user is looking on screen",
    author="Refactored by Python Team",
    python_requires=">=3.8",
)
