import os
import hashlib
from PIL import Image

def generate_hash(image_path, hash_algo="md5"):
    """
    Generate a hash for the given image using the specified hashing algorithm.
    """
    image = Image.open(image_path)
    # Convert image to bytes
    image_bytes = image.tobytes()

    # Generate hash of the image bytes
    if hash_algo == "md5":
        return hashlib.md5(image_bytes).hexdigest()
    elif hash_algo == "sha256":
        return hashlib.sha256(image_bytes).hexdigest()

def find_duplicate_images(directory, hash_algo="md5"):
    """
    Find duplicate images in the directory by comparing image hashes.
    """
    image_hashes = {}
    duplicate_images = []

    # Walk through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Only process image files (you can extend this to check for more formats)
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_hash = generate_hash(file_path, hash_algo)

            # If the hash already exists, it's a duplicate
            if file_hash in image_hashes:
                duplicate_images.append((image_hashes[file_hash], file_path))
            else:
                image_hashes[file_hash] = file_path

    return duplicate_images

# Example usage:
directory_path = "dataset/yolo_valid/images"  # Update with your directory
duplicates = find_duplicate_images(directory_path)

if duplicates:
    print("Found duplicate images:")
    for dup in duplicates:
        print(f"Duplicate: {dup[0]} and {dup[1]}")
else:
    print("No duplicates found.")
