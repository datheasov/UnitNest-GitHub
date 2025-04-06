import os
import subprocess

# Set up the environment
os.environ["PYTHONPATH"] = "D:/A_Capstone1/UnitNest_MeterDetection_Yolov8_ptv3/PaddleOCR"

# Paths to training script, config file, and model directories
paddleocr_train_script = "D:/A_Capstone1/UnitNest_MeterDetection_Yolov8_ptv3/PaddleOCR/tools/train.py"
config_file = r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\configs\rec\ch_PP-OCRv3_det_cml.yml"

# Use the fine-tuning model from the distillation-trained checkpoint
pretrained_model = "D:/A_Capstone1/UnitNest_MeterDetection_Yolov8_ptv3/pre_train_model/en_PP-OCRv3_det_distill_train/best_accuracy"

# Directory where the fine-tuned model will be saved
save_model_dir = "D:/A_Capstone1/UnitNest_MeterDetection_Yolov8_ptv3/PaddleOCR/outs5/rec_en_number_finetuned"

# Construct the PaddleOCR fine-tuning command
command = [
    "python", paddleocr_train_script,
    "-c", config_file,
    "-o", f'Global.pretrained_model="{pretrained_model}"',
    "-o", f'Global.save_model_dir="{save_model_dir}"'
]

# Execute the training command
try:
    subprocess.run(command, check=True)
    print("✅ Fine-tuning started successfully!")
except subprocess.CalledProcessError as e:
    print(f"❌ Error during training: {e}")
