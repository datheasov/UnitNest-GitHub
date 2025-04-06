import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_file = "train.log"

# === Step 1: Debug - Print all lines containing 'epoch' (case insensitive) ===
print("=== Lines containing 'epoch' in train.log ===")
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        if "epoch" in line.lower():
            print(line.strip())

# === Step 2: Define a flexible regex to capture training metrics ===
# This pattern is designed to catch lines that include epoch, loss, acc, and optionally
# precision, recall, f1, wer, and avg levenshtein (with several common variations).
pattern = re.compile(
    r"epoch[:\s]*(\d+).*?loss[:=]\s*([\d\.]+).*?acc(?:uracy)?[:=]\s*([\d\.]+)"
    r"(?:.*?prec(?:ision)?[:=]\s*([\d\.]+))?"
    r"(?:.*?rec(?:all)?[:=]\s*([\d\.]+))?"
    r"(?:.*?f1(?:\s*score)?[:=]\s*([\d\.]+))?"
    r"(?:.*?wer[:=]\s*([\d\.]+))?"
    r"(?:.*?(?:avg[\s_-]*lev(?:enshtein)?)[:=]\s*([\d\.]+))?",
    re.IGNORECASE
)

# Initialize lists for each metric
epochs = []
losses = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
wers = []
avg_levenshtein = []

# === Step 3: Read the log file and extract metrics using the regex ===
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            accuracies.append(float(match.group(3)))
            # For optional groups, assign NaN if not present
            precisions.append(float(match.group(4)) if match.group(4) else np.nan)
            recalls.append(float(match.group(5)) if match.group(5) else np.nan)
            f1_scores.append(float(match.group(6)) if match.group(6) else np.nan)
            wers.append(float(match.group(7)) if match.group(7) else np.nan)
            avg_levenshtein.append(float(match.group(8)) if match.group(8) else np.nan)

if not epochs:
    print("No performance metrics were extracted. Please verify that your train.log contains training metrics and adjust the regex accordingly.")
else:
    # Create a DataFrame to display the metrics
    df = pd.DataFrame({
        "Epoch": epochs,
        "Loss": losses,
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
        "F1 Score": f1_scores,
        "WER": wers,
        "Avg Levenshtein Dist": avg_levenshtein
    })

    print("\n📊 **Extracted Training Performance Metrics** 📊\n")
    print(df)

    # Plot the metrics using matplotlib
    plt.figure(figsize=(18, 12))

    # Plot Loss over Epochs
    plt.subplot(3, 3, 1)
    plt.plot(df["Epoch"], df["Loss"], marker="o", linestyle="-", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # Plot Accuracy over Epochs
    plt.subplot(3, 3, 2)
    plt.plot(df["Epoch"], df["Accuracy"], marker="o", linestyle="-", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)

    # Plot Precision over Epochs
    plt.subplot(3, 3, 3)
    plt.plot(df["Epoch"], df["Precision"], marker="o", linestyle="-", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision")
    plt.grid(True)

    # Plot Recall over Epochs
    plt.subplot(3, 3, 4)
    plt.plot(df["Epoch"], df["Recall"], marker="o", linestyle="-", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall")
    plt.grid(True)

    # Plot F1 Score over Epochs
    plt.subplot(3, 3, 5)
    plt.plot(df["Epoch"], df["F1 Score"], marker="o", linestyle="-", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score")
    plt.grid(True)

    # Plot WER over Epochs
    plt.subplot(3, 3, 6)
    plt.plot(df["Epoch"], df["WER"], marker="o", linestyle="-", color="brown")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.title("Word Error Rate (WER)")
    plt.grid(True)

    # Plot Average Levenshtein Distance over Epochs
    plt.subplot(3, 3, 7)
    plt.plot(df["Epoch"], df["Avg Levenshtein Dist"], marker="o", linestyle="-", color="magenta")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Levenshtein Dist")
    plt.title("Average Levenshtein Distance")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
