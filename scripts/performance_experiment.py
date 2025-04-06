import json
import Levenshtein

def load_ground_truth_ordered(label_file):
    """
    Load ground truth meter numbers from a text file.
    Each line is expected to be in the format:
       e1.jpg 03258
       e2.jpg 01150
       ...
    The order of lines is assumed to correspond to your test set order.
    Returns a list of meter numbers.
    """
    gt_list = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # We ignore the filename and keep the meter number.
                _, meter_number = parts
                gt_list.append(meter_number)
    return gt_list

def load_experiment_results_ordered(json_file):
    """
    Load OCR experiment results from a JSON file.
    Assumes the JSON array order corresponds to the order of ground truth labels.
    Each element should be a dictionary that includes the key "Meter Number".
    For example:
    [
       {"Meter Type": "Electricity", "Meter Number": "03258", "Accuracy": "98.12%"},
       {"Meter Type": "Electricity", "Meter Number": "01150", "Accuracy": "99.87%"},
       ...
    ]
    """
    with open(json_file, "r") as f:
        results = json.load(f)
    return results

def evaluate_ocr_ordered(gt_list, results):
    total = len(gt_list)
    exact_matches = 0
    total_distance = 0
    total_wer = 0.0  # Accumulate per-sample WER (%)
    total_confidence = 0.0  # Accumulate confidence scores
    count_conf = 0         # Count of results with a valid confidence score
    total_chars = 0        # Total number of characters in all ground truth labels

    for i, true_meter in enumerate(gt_list):
        result = results[i]
        predicted = result.get("Meter Number", "N/A")
        confidence_str = result.get("Accuracy", "0%")
        try:
            confidence = float(confidence_str.replace("%", ""))
        except:
            confidence = 0.0

        if predicted == true_meter:
            exact_matches += 1

        # Compute Levenshtein distance
        distance = Levenshtein.distance(predicted, true_meter)
        total_distance += distance

        # Compute WER (Character Error Rate) per sample: (distance / length) * 100
        if len(true_meter) > 0:
            wer = (distance / len(true_meter)) * 100
        else:
            wer = 0
        total_wer += wer

        total_confidence += confidence
        count_conf += 1
        total_chars += len(true_meter)

    exact_accuracy = (exact_matches / total * 100) if total > 0 else 0
    avg_distance = (total_distance / total) if total > 0 else 0
    avg_wer = (total_wer / total) if total > 0 else 0
    overall_cer = (total_distance / total_chars * 100) if total_chars > 0 else 0
    avg_confidence = (total_confidence / count_conf) if count_conf > 0 else 0

    print("OCR Performance Metrics:")
    print("========================")
    print(f"  - Total Images Processed: {total}")
    print(f"  - Exact Match Accuracy: {exact_accuracy:.2f}%")
    print(f"  - Average Levenshtein Distance: {avg_distance:.2f}")
    print(f"  - Average WER (Per-sample Character Error Rate): {avg_wer:.2f}%")
    print(f"  - Overall CER (Total Character Error Rate): {overall_cer:.2f}%")
    print(f"  - Average Confidence Score: {avg_confidence:.2f}%")
    return exact_accuracy, avg_distance, avg_wer, overall_cer, avg_confidence

if __name__ == "__main__":
    # Update these file paths to point to your actual files.
    ground_truth_file = r"D:\A_Capstone1\UnitNest_MeterDetection_Yolov8_ptv3\Testing\label.txt"
    experiment_results_file = r"./Result_experiment/v5810_10075100GF_Edge_final2.json"  # Your OCR output saved as JSON

    gt_list = load_ground_truth_ordered(ground_truth_file)
    results = load_experiment_results_ordered(experiment_results_file)
    evaluate_ocr_ordered(gt_list, results)
