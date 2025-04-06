import os
import cv2 
import numpy as np

def preprocess_image_for_cnn(image_path, img_size=224):
    """ 
    Preprocess an image for CNN classification:
    - Convert to grayscale (if needed)
    - Resize to a fixed size (224x224)
    - Normalize pixel values (0-255 → 0-1)
    """
    image = cv2.imread(image_path)  # Read the image
    if image is None:
        print(f"❌ Skipping invalid image: {image_path}")
        return None

    # Convert to grayscale (if needed)
    if len(image.shape) == 3:  # If RGB, convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to ensure uniform input size
    image = cv2.resize(image, (img_size, img_size))

    # Normalize pixel values to range [0,1]
    image = np.expand_dims(image, axis=-1)  # Ensure shape (224, 224, 1)
    image = image.astype(np.float32) / 255.0  

    return image

def process_all_folders(base_folder, output_base="preprocessed_cnn"):
    """ Processes all images in 'cnn_train', 'cnn_valid', and 'cnn_test' datasets. """
    os.makedirs(output_base, exist_ok=True)
    
    for dataset_type in ["cnn_train", "cnn_valid", "cnn_test"]:  # Process all datasets
        dataset_path = os.path.join(base_folder, dataset_type)
        output_dataset_path = os.path.join(output_base, dataset_type)
        os.makedirs(output_dataset_path, exist_ok=True)

        for category in ["water", "electricity"]:  # Process both categories
            category_path = os.path.join(dataset_path, category)
            output_category_path = os.path.join(output_dataset_path, category)
            os.makedirs(output_category_path, exist_ok=True)

            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(category_path, filename)
                    cnn_image = preprocess_image_for_cnn(image_path)

                    if cnn_image is not None:
                        save_path = os.path.join(output_category_path, filename)
                        
                        # Convert back to uint8 before saving
                        processed_image = (cnn_image * 255).astype(np.uint8)
                        cv2.imwrite(save_path, processed_image)
                        
                        print(f"✅ Processed and saved: {save_path}")

if __name__ == "__main__":
    dataset_folder = "dataset"  # Adjust path if needed
    process_all_folders(dataset_folder)