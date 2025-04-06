import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- CONFIGURATION ----------------------
CNN_TRAIN_DIR = "preprocessed_cnn/cnn_train"
CNN_VALID_DIR = "preprocessed_cnn/cnn_valid"
CNN_TEST_DIR = "preprocessed_cnn/cnn_test"
MODEL_SAVE_PATH = "models/resnet_gray"  # Save path for ResNet model

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# ---------------------- CNN IMAGE PREPROCESSING ----------------------
print("\n🔄 Preparing Image Data Generators...")
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    CNN_TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="rgb"
)
valid_generator = valid_test_datagen.flow_from_directory(
    CNN_VALID_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="rgb"
)
test_generator = valid_test_datagen.flow_from_directory(
    CNN_TEST_DIR, target_size=(224, 224), batch_size=32, class_mode="binary", color_mode="rgb", shuffle=False
)
print("✅ Data Preparation Completed!\n")

# ---------------------- LOAD OR CREATE RESNET MODEL ----------------------
if os.path.exists(MODEL_SAVE_PATH):
    print("\n🔄 Loading existing ResNet model...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
else:
    print("\n🚀 Creating a new ResNet model...")
    
    # Load Pretrained ResNet50 without the top classification layer
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # Freeze pretrained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = Flatten()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)  # Binary classification (Meter reading detection)

    model = Model(inputs=base_model.input, outputs=x)

    # Compile Model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("✅ ResNet model created successfully!")

# ---------------------- EVALUATE RESNET MODEL ----------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"📊 Final ResNet Test Accuracy: {test_acc * 100:.2f}%")

# ---------------------- ADDITIONAL PERFORMANCE METRICS ----------------------
# Predict labels for the test set
test_labels = test_generator.classes  # True labels
predictions = model.predict(test_generator, verbose=1)
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
plt.title('Confusion Matrix - ResNet')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
