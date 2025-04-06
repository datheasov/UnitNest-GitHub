import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# ---------------------- CONFIGURATION ----------------------
CNN_TRAIN_DIR = "preprocessed_cnn/cnn_train"  # Path to your training data (grayscale images)
CNN_VALID_DIR = "preprocessed_cnn/cnn_valid"  # Path to your validation data
CNN_TEST_DIR  = "preprocessed_cnn/cnn_test"   # Path to your test data
CNN_MODEL_SAVE_PATH = "models/resnet_gray3"  # Path to save the trained model

# Ensure necessary directories exist
os.makedirs("models", exist_ok=True)

# ---------------------- IMAGE PREPROCESSING ----------------------
print("\n🔄 Preparing Image Data Generators...")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Only normalization for validation and testing
valid_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Set color_mode="grayscale" since your preprocessed images are already grayscale
train_generator = train_datagen.flow_from_directory(
    CNN_TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    color_mode="grayscale"
)

valid_generator = valid_test_datagen.flow_from_directory(
    CNN_VALID_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    color_mode="grayscale"
)

test_generator = valid_test_datagen.flow_from_directory(
    CNN_TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    color_mode="grayscale",
    shuffle=False
)

print("✅ Image Data Preparation Completed!\n")

# ---------------------- DEFINE RESNET50 MODEL ----------------------
print("\n🚀 Building ResNet50 Model for Meter Type Classification...")

# Create an input layer for grayscale images
input_layer = Input(shape=(224, 224, 1))
# Convert grayscale image to pseudo-RGB (3 channels) using a Lambda layer
rgb_layer = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(input_layer)

# Load pre-trained ResNet50 with ImageNet weights using the pseudo-RGB input
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=rgb_layer)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

cnn_model_resnet = Model(inputs=input_layer, outputs=output)

# Freeze the base model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
cnn_model_resnet.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ---------------------- CALLBACKS ----------------------
checkpoint = ModelCheckpoint(
    CNN_MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
callbacks = [checkpoint, reduce_lr]

# ---------------------- TRAIN RESNET50 MODEL ----------------------
print("\n🚀 Training ResNet50 Model...")
history = cnn_model_resnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=100,  # Number of epochs
    callbacks=callbacks
)
print("✅ ResNet50 Model Training Completed!\n")

# ---------------------- SAVE RESNET50 MODEL ----------------------
cnn_model_resnet.save(CNN_MODEL_SAVE_PATH, save_format="tf")
print(f"✅ ResNet50 Model Saved at {CNN_MODEL_SAVE_PATH}")

# ---------------------- EVALUATE RESNET50 MODEL ----------------------
test_loss, test_acc = cnn_model_resnet.evaluate(test_generator)
print(f"📊 Final ResNet50 Test Accuracy: {test_acc * 100:.2f}%")
