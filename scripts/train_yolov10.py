import os
import glob
import cv2
import yaml
from ultralytics import YOLO

# ======= CONFIGURATION =======
DATA_PATH = r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\dataset\config.yaml"  # <-- Dataset YAML path
MODEL_PATH = "yolov10/weights/yolov10n.pt"  # <-- Pretrained weights

EPOCHS = 100        # Training epochs
BATCH_SIZE = 16     # Adjust for your GPU
IMG_SIZE = 640      # Image size
DEVICE = "cuda"     # "cuda" for GPU, "cpu" otherwise

# Save directory
PROJECT_DIR = "runsv10_gray3/train"  # Main directory for experiments
EXPERIMENT_NAME = "custom_yolov10"   # Unique experiment name

# ======= FUNCTIONS =======
def convert_images_to_grayscale(directory):
    """
    Convert all images in the specified directory to grayscale.
    The image is read, converted to grayscale, then converted back to a 3-channel image 
    to ensure compatibility with YOLOv10.
    """
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    for ext in image_extensions:
        for filepath in glob.glob(os.path.join(directory, ext)):
            img = cv2.imread(filepath)
            if img is None:
                continue  # Skip unreadable files
            # Convert to grayscale then back to 3 channels (BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(filepath, gray_bgr)
            print(f"Converted {filepath} to grayscale.")

def preprocess_dataset(data_config_path):
    """
    Load the dataset YAML configuration to locate the training and validation directories,
    then convert the images in those directories to grayscale.
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
    # ======= PREPROCESSING =======
    print("\n🔄 Preprocessing dataset: Converting images to grayscale...")
    preprocess_dataset(DATA_PATH)

    # ======= LOAD YOLOV10 MODEL =======
    print("\n🚀 Loading YOLOv10 model...")
    model = YOLO(MODEL_PATH)  # Load pre-trained YOLOv10

    # ======= TRAINING =======
    print("\n📢 Starting YOLOv10 training...")
    model.train(
        data=DATA_PATH,       # Dataset YAML file
        epochs=EPOCHS,        # Number of epochs
        batch=BATCH_SIZE,     # Batch size
        imgsz=IMG_SIZE,       # Image size
        device=DEVICE,        # Device: GPU or CPU
        project=PROJECT_DIR,  # Save under 'runs/train'
        name=EXPERIMENT_NAME, # Name of this experiment
        pretrained=True,      # Use pretrained model

        # ✅ Optimized Data Augmentation
        hsv_h=0.015,      # Hue augmentation
        hsv_s=0.7,        # Saturation augmentation
        hsv_v=0.4,        # Brightness augmentation
        scale=0.5,        # Random scaling
        translate=0.1,    # Random translation
        flipud=0.2,       # Vertical flip probability
        fliplr=0.5,       # Horizontal flip probability
        mosaic=1.0,       # Mosaic augmentation (useful for small datasets)
        mixup=0.2,        # Mixup augmentation

        workers=4,        # Number of data loading workers (adjust for speed)
    )

    # ======= RESULTS & SAVE PATH =======
    best_model_path = f"{PROJECT_DIR}/{EXPERIMENT_NAME}/weights/best.pt"
    print("\n✅ Training completed successfully!")
    print(f"📂 Best model saved at: {best_model_path}")

    # ======= TENSORBOARD INSTRUCTIONS =======
    print("\nℹ️ To monitor training progress, run:")
    print(f"    tensorboard --logdir {PROJECT_DIR}")
    
    print("\n🎉 YOLOv10 Training Finished!")
