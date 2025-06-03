# STEP 1: Install all required packages
!pip install tensorflow scikit-learn seaborn matplotlib numpy streamlit streamlit-drawable-canvas opencv-python Pillow pyngrok

# STEP 2: First, run your training code to create the model
# This is based on your Rajiya_CNN.py file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🚀 Starting model training...")

# Load and preprocess MNIST data
print("📁 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for CNN
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert labels to categorical
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("✅ Data preprocessing completed!")

# Build CNN model
print("🧠 Building CNN model...")
model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("📋 Model architecture:")
model.summary()

# Train the model (reduced epochs for faster training)
print("🏃‍♂️ Training model (this may take a few minutes)...")

# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=3, 
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, 
        patience=2,
        monitor='val_loss'
    )
]

# Train the model
history = model.fit(
    x_train, y_train_cat,
    batch_size=128,
    epochs=10,  # Reduced for faster training
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
print("📊 Evaluating model...")
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

# Save the model
model_filename = 'mnist_digit_recognition_model.h5'
model.save(model_filename)
print(f"💾 Model saved as {model_filename}")

# Verify the file was created
import os
if os.path.exists(model_filename):
    print(f"✅ Model file created successfully!")
    print(f"📁 File size: {os.path.getsize(model_filename) / (1024*1024):.1f} MB")
else:
    print("❌ Error: Model file was not created!")

print("\n" + "="*50)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")
print("🚀 Ready to run Streamlit app!")
print("="*50)