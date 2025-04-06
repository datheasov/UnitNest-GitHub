import subprocess
import os
import glob
import cv2
import yaml

# ---------------------- CONFIGURATION ----------------------
WEIGHTS = "yolov5s.pt"  # Pre-trained YOLOv5s weights
DATA_CONFIG = r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\dataset\config.yaml"  # Dataset YAML configuration
EPOCHS = 100         # Number of training epochs
IMAGE_SIZE = 640     # Input image size in pixels
BATCH_SIZE = 4       # Batch size (adjust if needed for memory constraints)
PROJECT_DIR = "runsv5_gray3/train"
EXPERIMENT_NAME = "exp_yolov5_training_aug"

# ---------------------- DATA AUGMENTATION CONFIGURATION ----------------------
HYP_FILE = r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\yolov5\data\hyps\hyp.scratch-low.yaml"

# ---------------------- FUNCTIONS ----------------------
def convert_images_to_grayscale(directory):
    """
    Finds image files in the given directory, converts them to grayscale,
    then converts back to 3 channels (so they remain compatible with YOLOv5).
    """
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    for ext in image_extensions:
        for filepath in glob.glob(os.path.join(directory, ext)):
            img = cv2.imread(filepath)
            if img is None:
                continue  # Skip files that cannot be read
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Convert back to BGR (3 channels) so the model input shape is preserved
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(filepath, gray_bgr)
            print(f"Converted {filepath} to grayscale.")

# ---------------------- PREPROCESSING: Convert images to grayscale ----------------------
# Load dataset configuration YAML to extract training and validation directories
with open(DATA_CONFIG, 'r') as file:
    data_cfg = yaml.safe_load(file)

# It is assumed that your YAML config contains 'train' and 'val' keys with paths.
train_dir = data_cfg.get('train')
val_dir = data_cfg.get('val')

if train_dir and os.path.isdir(train_dir):
    print("Converting training images to grayscale...")
    convert_images_to_grayscale(train_dir)
else:
    print("Training directory not found or not specified in the config.")

if val_dir and os.path.isdir(val_dir):
    print("Converting validation images to grayscale...")
    convert_images_to_grayscale(val_dir)
else:
    print("Validation directory not found or not specified in the config.")

# ---------------------- TRAINING COMMAND ----------------------
command = [
    "python", "train.py",
    "--weights", WEIGHTS,
    "--data", DATA_CONFIG,
    "--epochs", str(EPOCHS),
    "--img", str(IMAGE_SIZE),
    "--batch", str(BATCH_SIZE),
    "--project", PROJECT_DIR,
    "--name", EXPERIMENT_NAME,
    "--hyp", HYP_FILE  # Enables data augmentation from the hyperparameter file
]

print("🚀 Starting YOLOv5 training with grayscale images and data augmentation...")
subprocess.run(command, cwd=r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\yolov5")
print("✅ YOLOv5 training completed!")
