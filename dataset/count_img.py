import os

def count_images(directory, extensions=[".jpg", ".jpeg", ".png", ".bmp", ".tiff",".txt"]):
    """
    Count the number of images in a given directory (and subdirectories).
    
    Args:
        directory (str): Path to the directory containing images.
        extensions (list): List of image file extensions to consider.
        
    Returns:
        int: The number of images found in the directory.
    """
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter the files based on extensions
    image_files = [f for f in files if any(f.endswith(ext) for ext in extensions)]
    
    # Return the count of image files
    return len(image_files)

# Specify the directory you want to count images in
directory_path = "dataset/yolo_valid/labels"  # Update to your directory

# Count and print the number of images
num_images = count_images(directory_path)
print(f"Total number of images in {directory_path}: {num_images}")
