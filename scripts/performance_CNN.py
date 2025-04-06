import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- CONFIGURATION ----------------------
CNN_TRAIN_DIR = "preprocessed_cnn/cnn_train"
CNN_VALID_DIR = "preprocessed_cnn/cnn_valid"
CNN_TEST_DIR = "preprocessed_cnn/cnn_test"
CNN_MODEL_SAVE_PATH = "models/cnn_gray2"  # Save in TensorFlow format

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# ---------------------- CNN IMAGE PREPROCESSING ----------------------
print("\n🔄 Preparing CNN Image Data Generators...")
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    CNN_TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="grayscale"
)
valid_generator = valid_test_datagen.flow_from_directory(
    CNN_VALID_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="grayscale"
)
test_generator = valid_test_datagen.flow_from_directory(
    CNN_TEST_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="grayscale", shuffle=False
)
print("✅ CNN Data Preparation Completed!\n")

# ---------------------- LOAD EXISTING CNN MODEL ----------------------
if os.path.exists(CNN_MODEL_SAVE_PATH):
    print("\n🔄 Loading existing CNN model...")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_SAVE_PATH)
else:
    print(f"⚠️ No model found at {CNN_MODEL_SAVE_PATH}. Please train and save the model first.")
    exit(1)

# ---------------------- EVALUATE CNN MODEL ----------------------
test_loss, test_acc = cnn_model.evaluate(test_generator)
print(f"📊 Final CNN Test Accuracy: {test_acc * 100:.2f}%")

# ---------------------- ADDITIONAL PERFORMANCE METRICS ----------------------
# Predict labels for the test set
test_labels = test_generator.classes  # True labels
predictions = cnn_model.predict(test_generator, verbose=1)
predicted_labels = (predictions > 0.5).astype("int32")  # Convert predictions to 0 or 1

# Calculate Precision, Recall, F1-Score
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

# Calculate Confusion Matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Print additional metrics
print(f"📊 Precision: {precision * 100:.2f}%")
print(f"📊 Recall: {recall * 100:.2f}%")
print(f"📊 F1-Score: {f1 * 100:.2f}%")
print(f"📊 Confusion Matrix:\n{cm}")

# ---------------------- PLOT CONFUSION MATRIX ----------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# import numpy as np

# # ---------------------- CONFIGURATION ----------------------
# CNN_TRAIN_DIR = "preprocessed_cnn/cnn_train"
# CNN_VALID_DIR = "preprocessed_cnn/cnn_valid"
# CNN_TEST_DIR = "preprocessed_cnn/cnn_test"
# CNN_MODEL_SAVE_PATH = "models/meter_classifier"  # Save in TensorFlow format

# # Ensure necessary directories exist
# os.makedirs("models", exist_ok=True)

# # ---------------------- CNN IMAGE PREPROCESSING ----------------------
# print("\n🔄 Preparing CNN Image Data Generators...")
# train_datagen = ImageDataGenerator(rescale=1.0 / 255)
# valid_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# train_generator = train_datagen.flow_from_directory(
#     CNN_TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="grayscale"
# )
# valid_generator = valid_test_datagen.flow_from_directory(
#     CNN_VALID_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="grayscale"
# )
# test_generator = valid_test_datagen.flow_from_directory(
#     CNN_TEST_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="grayscale", shuffle=False
# )
# print("✅ CNN Data Preparation Completed!\n")

# # ---------------------- LOAD EXISTING CNN MODEL ----------------------
# if os.path.exists(CNN_MODEL_SAVE_PATH):
#     print("\n🔄 Loading existing CNN model...")
#     cnn_model = tf.keras.models.load_model(CNN_MODEL_SAVE_PATH)
# else:
#     print(f"⚠️ No model found at {CNN_MODEL_SAVE_PATH}. Please train and save the model first.")
#     exit(1)

# # ---------------------- EVALUATE CNN MODEL ----------------------
# test_loss, test_acc = cnn_model.evaluate(test_generator)
# print(f"📊 Final CNN Test Accuracy: {test_acc * 100:.2f}%")

# # ---------------------- ADDITIONAL PERFORMANCE METRICS ----------------------
# # Predict labels for the test set
# test_labels = test_generator.classes  # True labels
# predictions = cnn_model.predict(test_generator, verbose=1)
# predicted_labels = (predictions > 0.5).astype("int32")  # Convert predictions to 0 or 1

# # Calculate Precision, Recall, F1-Score
# precision = precision_score(test_labels, predicted_labels)
# recall = recall_score(test_labels, predicted_labels)
# f1 = f1_score(test_labels, predicted_labels)

# # Calculate Confusion Matrix
# cm = confusion_matrix(test_labels, predicted_labels)

# # Print additional metrics
# print(f"📊 Precision: {precision * 100:.2f}%")
# print(f"📊 Recall: {recall * 100:.2f}%")
# print(f"📊 F1-Score: {f1 * 100:.2f}%")
# print(f"📊 Confusion Matrix:\n{cm}")
