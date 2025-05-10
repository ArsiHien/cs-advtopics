# Advanced topics in Computer Science

This repository contains the implementation of a gaze tracking pipeline, sourced from [pperle/gaze-tracking-pipeline](https://github.com/pperle/gaze-tracking-pipeline).

## Installation and Setup

Follow these steps to set up and run the gaze tracking pipeline on your local machine.

### Prerequisites
- Python 3.12

### Setup Instructions

1. **Create a Virtual Environment**
   ```bash
   python -m venv myenv
   ```

2. **Activate the Virtual Environment**
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS or Linux:
     ```bash
     source myenv/bin/activate
     ```

3. **Install Requirements**
   Ensure you are in the project directory and the virtual environment is activated. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Pipeline**
   Execute the main script with the path to the model checkpoint:
   ```bash
   python main.py --model_path="p00.ckpt"
   ```

## Notes
- Ensure the `p00.ckpt` model file is available in the project directory or provide the correct path to the model.
- Additional configuration options may be available in `main.py`. Refer to the script or the original repository for further details.

For more information, visit the [original repository](https://github.com/pperle/gaze-tracking-pipeline).
