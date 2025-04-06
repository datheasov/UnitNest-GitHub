import os
import glob
import cv2
import yaml
from ultralytics import YOLO

# ---------------------- CONFIGURATION ----------------------
MODEL_NAME = "yolov8s.pt"              # Pre-trained YOLOv8 small model
DATA_CONFIG = r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\dataset\config.yaml"    # Path to dataset configuration file
EPOCHS = 100                            # Number of training epochs
IMAGE_SIZE = 640                       # Input image size (pixels)
BATCH_SIZE = 16                        # Adjust based on your GPU capacity
DEVICE = "cuda"                        # Use GPU if available

# Experiment tracking settings
PROJECT_DIR = "runsv8_gray3/train"            # Directory to save training results
EXPERIMENT_NAME = "exp_yolov8_training"  # Name of this experiment

# ---------------------- FUNCTIONS ----------------------
def convert_images_to_grayscale(directory):
    """
    Convert all images in the specified directory to grayscale.
    Images are read, converted to grayscale, then converted back to a 3-channel image 
    so that the model's expected input shape is preserved.
    """
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    for ext in image_extensions:
        for filepath in glob.glob(os.path.join(directory, ext)):
            img = cv2.imread(filepath)
            if img is None:
                continue  # Skip files that cannot be read
            # Convert to grayscale and back to 3 channels
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(filepath, gray_bgr)
            print(f"Converted {filepath} to grayscale.")

def preprocess_dataset(data_config_path):
    """
    Load the dataset YAML configuration to locate training and validation directories,
    then convert images in these directories to grayscale.
    """
    with open(data_config_path, 'r') as file:
        data_cfg = yaml.safe_load(file)

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

if __name__ == '__main__':  # Prevent multiprocessing issues on Windows
    # ---------------------- PREPROCESSING ----------------------
    print("\n🔄 Preprocessing dataset: Converting images to grayscale...")
    preprocess_dataset(DATA_CONFIG)

    # ---------------------- MODEL INITIALIZATION ----------------------
    print("\n🚀 Loading YOLOv8 model...")
    model = YOLO(MODEL_NAME)  # Load the pre-trained YOLOv8 model

    # ---------------------- TRAINING ----------------------
    print(f"\n📢 Starting training on dataset: {DATA_CONFIG}")
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        
        # DATA AUGMENTATION SETTINGS
        hsv_h=0.015,      # Hue augmentation
        hsv_s=0.7,        # Saturation augmentation
        hsv_v=0.4,        # Brightness augmentation
        scale=0.5,        # Random scaling
        translate=0.1,    # Random translation (shifting)
        flipud=0.2,       # Random vertical flip
        fliplr=0.5,       # Random horizontal flip
        mosaic=1.0,       # Enable mosaic augmentation
        mixup=0.2,        # Enable mixup augmentation

        workers=0,        # For Windows: set workers=0 to prevent multiprocessing issues
    )

    # ---------------------- VALIDATION ----------------------
    print("\n📊 Evaluating the trained model on the validation set...")
    val_metrics = model.val(data=DATA_CONFIG)
    print("\nValidation Metrics:")
    print(val_metrics)
    
    # ---------------------- RESULTS AND SAVING ----------------------
    best_model_path = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "weights", "best.pt")
    print("\n✅ Training completed successfully!")
    print(f"📂 Best model saved at: {best_model_path}")

    # ---------------------- TENSORBOARD INSTRUCTIONS ----------------------
    print("\nℹ️ To monitor training and validation metrics, launch TensorBoard with:")
    print(f"    tensorboard --logdir {PROJECT_DIR}")
    
    print("\n🎉 YOLO Training Process Completed!")
