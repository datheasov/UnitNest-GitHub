import os
import cv2
import numpy as np
import tensorflow as tf
import re
import easyocr  # Using EasyOCR instead of PaddleOCR
from ultralytics import YOLO

# =================== CONFIGURATION ===================
SAVE_IMAGES = True
OUTPUT_DIR = "output_images"
USE_ADAPTIVE = True  # Set to False to disable adaptive thresholding if needed

# =================== GPU SETUP ===================
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ TensorFlow is using GPU!")
    except Exception as e:
        print("❌ TensorFlow GPU Error:", str(e))

# =================== LOAD MODELS ===================

# --- Load CNN Model for Meter Classification ---
try:
    cnn_model = tf.keras.models.load_model("models/meter_classifier2")
    print("✅ CNN model loaded successfully!")
except Exception as e:
    print("❌ CNN Model Loading Error:", str(e))
    exit()

# --- Load YOLO Models (v5, v8, and v10) ---
try:
    yolo_v5 = YOLO("yolov5s.pt")
    yolo_v8 = YOLO(r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\runsv8_gray2\train\exp_yolov8_training\weights\best.pt")
    yolo_v10 = YOLO(r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\runsv10_gray3\train\custom_yolov10\weights\best.pt")
    yolo_v5.to("cuda")
    yolo_v8.to("cuda")
    yolo_v10.to("cuda")
    print("✅ YOLOv5, YOLOv8, and YOLOv10 models loaded successfully on GPU!")
except Exception as e:
    print("❌ YOLO Models Loading Error:", str(e))
    exit()

# --- Initialize EasyOCR Readers ---
# These three readers simulate the "special", "default", and "fine-tuned" models.
try:
    ocr_special = easyocr.Reader(['en'], gpu=True)
    print("✅ EasyOCR (Specialized) loaded successfully!")
    ocr_default = easyocr.Reader(['en'], gpu=True)
    print("✅ EasyOCR (Default) loaded successfully!")
    ocr_finetuned = easyocr.Reader(['en'], gpu=True)
    print("✅ EasyOCR (Fine-tuned) loaded successfully!")
except Exception as e:
    print("❌ EasyOCR Loading Error:", str(e))
    exit()

CLASS_LABELS = {0: "Electricity", 1: "Water"}

# =================== HELPER PREPROCESSING FUNCTIONS ===================
def ensure_min_size(img, min_width=300):
    h, w = img.shape[:2]
    if w < min_width:
        scale_factor = min_width / float(w)
        return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    return img

def denoise_image(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def clahe_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_adaptive(image):
    enhanced = clahe_enhancement(image)
    adapt = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    return adapt

def preprocess_denoised_clahe(image):
    denoised = denoise_image(image)
    return clahe_enhancement(denoised)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def morphological_processing(image):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def gamma_correction(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def otsu_threshold(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# =================== ADVANCED PREPROCESSING FUNCTIONS ===================
def super_resolution(image, scale=2):
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

def perspective_correction(image, pts=None):
    if pts is None:
        return image
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def multi_scale_crops(image, scales=[0.9, 1.0, 1.1]):
    """Generate additional crops at multiple scales."""
    h, w = image.shape[:2]
    crops = []
    for scale in scales:
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Center crop:
        x_start = max(0, (w - new_w) // 2)
        y_start = max(0, (h - new_h) // 2)
        crop = image[y_start:y_start+new_h, x_start:x_start+new_w]
        crop = ensure_min_size(crop)
        crops.append(crop)
    return crops

def advanced_preprocessing(image):
    variants = []
    # Original grayscale variant
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = ensure_min_size(gray)
    variants.append(("grayscale_adv", gray))
    # CLAHE enhanced
    clahe_img = clahe_enhancement(image)
    clahe_img = ensure_min_size(clahe_img)
    variants.append(("clahe_adv", clahe_img))
    # Adaptive thresholding
    adaptive_img = preprocess_adaptive(image)
    adaptive_img = ensure_min_size(adaptive_img)
    variants.append(("adaptive_adv", adaptive_img))
    # Denoised + CLAHE
    denoised_clahe_img = preprocess_denoised_clahe(image)
    denoised_clahe_img = ensure_min_size(denoised_clahe_img)
    variants.append(("denoised_clahe_adv", denoised_clahe_img))
    # Deskew, sharpen, morphological processing
    deskewed = deskew(gray)
    sharpened = sharpen_image(deskewed)
    morph_processed = morphological_processing(sharpened)
    morph_processed = ensure_min_size(morph_processed)
    variants.append(("deskew_sharpen_morph_adv", morph_processed))
    # Otsu thresholding
    otsu_img = otsu_threshold(gray)
    otsu_img = ensure_min_size(otsu_img)
    variants.append(("otsu_adv", otsu_img))
    # Gamma correction
    gamma_img = gamma_correction(gray, gamma=1.2)
    gamma_img = ensure_min_size(gamma_img)
    variants.append(("gamma_adv", gamma_img))
    # Super-resolution
    sr_gray = super_resolution(gray, scale=2)
    variants.append(("super_resolution_adv", sr_gray))
    return variants

def pipeline_variants_extended(image):
    """
    Combine standard preprocessing with advanced preprocessing and multi-scale cropping.
    """
    variants_dict = {}
    # Standard variants
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants_dict["grayscale"] = ensure_min_size(gray)
    variants_dict["clahe"] = ensure_min_size(clahe_enhancement(image))
    if USE_ADAPTIVE:
        variants_dict["adaptive"] = ensure_min_size(preprocess_adaptive(image))
    variants_dict["denoised_clahe"] = ensure_min_size(preprocess_denoised_clahe(image))
    variants_dict["otsu"] = ensure_min_size(otsu_threshold(gray))
    variants_dict["gamma"] = ensure_min_size(gamma_correction(gray, gamma=1.2))
    # Advanced variants
    adv_variants = advanced_preprocessing(image)
    for name, variant in adv_variants:
        key = name if name not in variants_dict else name + "_dup"
        variants_dict[key] = variant
    # Multi-scale crops
    multi_crops = multi_scale_crops(image)
    for idx, crop in enumerate(multi_crops):
        key = f"multiscale_{idx}"
        variants_dict[key] = ensure_min_size(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    return variants_dict

# =================== METER CLASSIFICATION ===================
def classify_meter(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Cannot read image {image_path}")
        return "Unknown"
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    prediction = cnn_model.predict(img)
    return CLASS_LABELS[int(prediction[0] > 0.5)]

# =================== OCR DIGIT EXTRACTION ===================
def postprocess_ocr_result(result):
    """
    Enforce that the OCR result is 4 or 5 digits. If not, return "N/A".
    Additional character corrections can be applied here.
    """
    if re.fullmatch(r'\d{4,5}', result):
        return result
    else:
        return "N/A"

def extract_4to5_digits(ocr_result, min_confidence=80.0):
    """
    Given an EasyOCR result (a list of tuples: (bbox, text, confidence)),
    return the best 4- or 5-digit candidate with confidence >= min_confidence.
    """
    if not ocr_result:
        return "N/A", 0.0
    candidates = []
    for bbox, text, conf in ocr_result:
        # Remove non-digit characters
        txt = re.sub(r'\D', '', text)
        if re.fullmatch(r'\d{4,5}', txt):
            conf_percent = conf * 100
            if conf_percent >= min_confidence:
                candidates.append((txt, conf_percent))
    if candidates:
        best_result = max(candidates, key=lambda x: x[1])
        return postprocess_ocr_result(best_result[0]), best_result[1]
    return "N/A", 0.0

# =================== YOLO DETECTION & OCR PIPELINE ===================
def get_combined_yolo_detections(original_img, conf_threshold=0.2):
    """Run YOLOv5, YOLOv8, and YOLOv10 on the image and combine their detections."""
    combined_detections = []
    for detector in [yolo_v5, yolo_v8, yolo_v10]:
        results = detector(original_img, conf=conf_threshold)
        if results and results[0].boxes and results[0].boxes.data is not None:
            detections = results[0].boxes.data.cpu().numpy()
            combined_detections.extend(detections)
    return combined_detections

def detect_and_extract_number(image_path):
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"❌ Error: Cannot read image {image_path}")
        return "N/A", 0

    if SAVE_IMAGES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        detected_boxes = get_combined_yolo_detections(original_img, conf_threshold=0.2)
        if not detected_boxes:
            print("❌ YOLO Detection Failed: No Results Found")
            return "N/A", 0

        print(f"✅ Combined YOLO detected {len(detected_boxes)} objects")
        best_meter_number = "N/A"
        best_confidence = 0.0
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        box_idx = 0

        for result in detected_boxes:
            box_idx += 1
            x1, y1, x2, y2, conf, cls_id = result.astype(int)
            if x2 <= x1 or y2 <= y1:
                print("⚠️ Skipping invalid bounding box.")
                continue

            box_width = x2 - x1
            padding = max(50, int(0.15 * box_width))
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(original_img.shape[1], x2 + padding)
            y2_padded = min(original_img.shape[0], y2 + padding)
            cropped = original_img[y1_padded:y2_padded, x1_padded:x2_padded]
            if cropped.size == 0:
                print("⚠️ Skipping empty cropped image.")
                continue

            if SAVE_IMAGES:
                crop_path = os.path.join(OUTPUT_DIR, f"{base_filename}_box{box_idx}_original.png")
                cv2.imwrite(crop_path, cropped)

            # Use extended pipeline variants including advanced and multi-scale crops
            variants = pipeline_variants_extended(cropped)
            for variant_name, preprocessed in variants.items():
                if len(preprocessed.shape) == 2:
                    gray_image = preprocessed
                    preprocessed_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                elif len(preprocessed.shape) == 3 and preprocessed.shape[2] == 3:
                    gray_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
                    preprocessed_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                else:
                    preprocessed_bgr = preprocessed

                if SAVE_IMAGES:
                    variant_crop_path = os.path.join(OUTPUT_DIR, f"{base_filename}_box{box_idx}_{variant_name}_gray.png")
                    cv2.imwrite(variant_crop_path, gray_image)

                # Run EasyOCR on this variant using three different reader instances
                ocr_result_special = ocr_special.readtext(preprocessed_bgr)
                text_special, conf_special = extract_4to5_digits(ocr_result_special)
                
                ocr_result_default = ocr_default.readtext(preprocessed_bgr)
                text_default, conf_default = extract_4to5_digits(ocr_result_default)
                
                ocr_result_finetuned = ocr_finetuned.readtext(preprocessed_bgr)
                text_finetuned, conf_finetuned = extract_4to5_digits(ocr_result_finetuned)
                
                # Combine the three results: choose the one with the highest confidence
                candidates = [
                    (text_special, conf_special),
                    (text_default, conf_default),
                    (text_finetuned, conf_finetuned)
                ]
                best_candidate = max(candidates, key=lambda x: x[1])
                text_found, conf_found = best_candidate

                if conf_found > best_confidence:
                    best_confidence = conf_found
                    best_meter_number = text_found

        return best_meter_number, round(best_confidence, 2)

    except Exception as e:
        print(f"❌ YOLO Processing Error: {str(e)}")
        return "N/A", 0

# =================== PROCESS MULTIPLE IMAGES ===================
def process_images(image_paths):
    results = []
    for image_path in image_paths:
        print(f"\n🔍 Processing: {image_path}")
        meter_type = classify_meter(image_path)
        meter_number, accuracy = detect_and_extract_number(image_path)
        results.append({
            "Meter Type": meter_type,
            "Meter Number": meter_number,
            "Accuracy": f"{accuracy}%"
        })
    return results

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    image_paths = [
        "Testing/e1.jpg", "Testing/e2.jpg", "Testing/e4.jpg", "Testing/e5.jpg",
        "Testing/e6.jpg", "Testing/e7.jpg", "Testing/e8.jpg", "Testing/e9.jpg", "Testing/e10.jpeg",
        "Testing/e11.jpeg", "Testing/e12.jpeg", "Testing/e13.jpg", "Testing/e14.jpg", "Testing/e15.jpg",
        "Testing/e16.jpg", "Testing/e17.jpeg", "Testing/e18.jpeg", "Testing/e19.jpeg", "Testing/e20.jpeg",
        "Testing/e21.jpeg", "Testing/e22.jpeg", "Testing/e23.jpeg", "Testing/e24.jpeg", "Testing/e25.jpeg",
        "Testing/e26.jpeg", "Testing/e27.jpeg", "Testing/e28.jpeg", "Testing/e29.jpeg", "Testing/e30.jpeg","Testing/e31.jpeg",
        "Testing/w1.jpg", "Testing/w2.jpg", "Testing/w3.jpg", "Testing/w4.jpg", "Testing/w5.jpg",
        "Testing/w6.jpg", "Testing/w7.jpg", "Testing/w8.jpeg", "Testing/w9.jpeg", "Testing/w10.jpg",
        "Testing/w11.jpg", "Testing/w12.jpg", "Testing/w13.jpg", "Testing/w14.jpg", "Testing/w15.jpg",
        "Testing/w16.jpg", "Testing/w17.jpg", "Testing/w18.jpg", "Testing/w19.jpeg", "Testing/w20.jpeg",
        "Testing/w21.jpeg","Testing/w22.jpeg","Testing/w23.jpeg","Testing/w24.jpeg","Testing/w25.jpeg",
        "Testing/w26.jpeg","Testing/w27.jpeg","Testing/w28.jpeg","Testing/w29.jpeg","Testing/w30.jpeg"
    ]
    output = process_images(image_paths)
    
    import json
    print("\n📌 Final Output:")
    print(json.dumps(output, indent=4))
    
    # Save the output to a JSON file
    with open("v5810_10075100GF_easyocr2.json", "w") as f:
        json.dump(output, f, indent=4)
