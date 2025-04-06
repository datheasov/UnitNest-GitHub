import os
import argparse

def validate_yolo_dataset(images_dir, labels_dir):
    """
    Validates a YOLO dataset by checking if each image has a corresponding label file.
    """
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png')

    # Get image files (without extensions)
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)}
    
    # Get label files (without extensions)
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

    # Find mismatches
    missing_labels = image_files - label_files  # Images without labels
    missing_images = label_files - image_files  # Labels without images

    # Print validation results
    print(f"\n🔍 Validating dataset: {images_dir}")
    print("-" * 50)

    if not image_files and not label_files:
        print("⚠️ Dataset folder is empty! Please check your dataset path.")
    else:
        if missing_labels:
            print(f"⚠️ {len(missing_labels)} images have NO corresponding label files:")
            for file in sorted(missing_labels):
                print(f"  ❌ Missing Label: {file}.txt")

        if missing_images:
            print(f"⚠️ {len(missing_images)} label files have NO corresponding images:")
            for file in sorted(missing_images):
                print(f"  ❌ Missing Image: {file}.jpg/.jpeg/.png")

        if not missing_labels and not missing_images:
            print("✅ Dataset validation passed: All images have corresponding labels.")

    print("-" * 50)

if __name__ == "__main__":
    # Allow command-line arguments for dataset path
    parser = argparse.ArgumentParser(description="YOLO Dataset Validator")
    parser.add_argument("--dataset", type=str, default="dataset", help="Base directory for YOLO dataset")

    args = parser.parse_args()
    base_folder = args.dataset  # Get dataset path from user input

    # Define YOLO dataset paths dynamically
    yolo_datasets = {
        "train": (os.path.join(base_folder, "yolo_train", "images"), os.path.join(base_folder, "yolo_train", "labels")),
        "valid": (os.path.join(base_folder, "yolo_valid", "images"), os.path.join(base_folder, "yolo_valid", "labels")),
        "test": (os.path.join(base_folder, "yolo_test", "images"), os.path.join(base_folder, "yolo_test", "labels"))
    }

    # Validate each dataset
    for dataset_name, (img_dir, lbl_dir) in yolo_datasets.items():
        if os.path.exists(img_dir) and os.path.exists(lbl_dir):  # Ensure paths exist
            validate_yolo_dataset(img_dir, lbl_dir)
        else:
            print(f"⚠️ Skipping {dataset_name} dataset: Folder not found!")