import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------- CONFIGURATION ----------------------
CNN_TRAIN_DIR = "preprocessed_cnn/cnn_train"
CNN_VALID_DIR = "preprocessed_cnn/cnn_valid"
CNN_TEST_DIR = "preprocessed_cnn/cnn_test"
CNN_MODEL_SAVE_PATH = "models/cnn_gray3"  # Save in TensorFlow format

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# ---------------------- CNN IMAGE PREPROCESSING & DATA AUGMENTATION ----------------------
print("\n🔄 Preparing CNN Image Data Generators...")

# Data augmentation for training (with extra brightness adjustment)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# Only normalization for validation and testing
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

# ---------------------- DEFINE CNN MODEL ----------------------
print("\n🚀 Building CNN Model for Meter Type Classification...")
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ---------------------- TRAIN CNN MODEL ----------------------
print("\n🚀 Training CNN Model...")
history = cnn_model.fit(train_generator, validation_data=valid_generator, epochs=100)
print("✅ CNN Model Training Completed!\n")

# ---------------------- SAVE CNN MODEL ----------------------
cnn_model.save(CNN_MODEL_SAVE_PATH, save_format="tf")  # Save in TensorFlow format
print(f"✅ CNN Model Saved at {CNN_MODEL_SAVE_PATH}")

# ---------------------- EVALUATE CNN MODEL ----------------------
test_loss, test_acc = cnn_model.evaluate(test_generator)
print(f"📊 Final CNN Test Accuracy: {test_acc * 100:.2f}%")
