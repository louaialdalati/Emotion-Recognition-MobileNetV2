"""
train_improved_v2.py

Hybrid Training Script for Colored Images (160x160):
- Phase 1: Feature Extraction (Train top layers only).
- Phase 2: Smart Fine-Tuning (Freeze first 100 layers, train rest).
- Features: Class Weights for Imbalance, MobileNetV2, Detailed Evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight # Important for handling imbalance
import seaborn as sns

# --- GPU Configuration ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.debugging.set_log_device_placement(False)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs detected: {gpus}")
    except RuntimeError as e:
        print(e)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Configuration ---
# UPDATED PATH: Pointing to the new colored images folder
train_dir = r'D:\projects\yazilim muhendisligi uygulamalari\face_detection\colored face images\train'
base_data_dir = os.path.dirname(train_dir)
val_dir = os.path.join(base_data_dir, 'validation')
test_dir = os.path.join(base_data_dir, 'test')

# UPDATED SIZE: 160x160 as agreed (Best compromise for upscaled data)
IMG_SIZE = (160, 160) 
BATCH_SIZE = 32 
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 8 
CLASSES = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

base_script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(base_script_dir, 'facial_expression_model.keras') 

# --- Data Preparation ---
def create_data_generators():
    print("Initializing generators...")
    print(f"Training Data Path: {train_dir}")
    
    # Moderate Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, 
        rotation_range=15,
        width_shift_range=0.15,  
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True
    )

    validation_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    
    # Test Generator
    if os.path.exists(test_dir):
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=CLASSES,
            shuffle=False
        )
    else:
        print("Warning: Test directory not found!")
        test_generator = None

    return train_generator, validation_generator, test_generator

# --- Model Building ---
def build_model(num_classes):
    print(f"Building MobileNetV2 model with input shape {IMG_SIZE}...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    
    # Freeze base model initially (Phase 1)
    base_model.trainable = False

    inputs = Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) 
    x = Dense(256, activation='relu')(x) 
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# --- Strategic Fine-Tuning ---
def unfreeze_model_smartly(model):
    """
    Unfreezes layers starting from index 100.
    """
    base_model = model.layers[1]
    base_model.trainable = True 
    
    # MobileNetV2 has 155 layers total.
    fine_tune_at = 100
    
    print(f"--- SMART FINE-TUNING: Freezing first {fine_tune_at} layers, training the rest ---")
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Re-compile with low learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Base model layers: {len(base_model.layers)}")
    print(f"Trainable layers: {len(model.trainable_variables)}")

# --- Callbacks ---
def create_callbacks():
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10, 
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        verbose=1,
        mode='min',
        min_lr=1e-7
    )
    return [checkpoint, early_stopping, reduce_lr]

# --- Plotting ---
def plot_training_history(history, stage_name=""):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{stage_name} Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{stage_name} Loss')

    plot_filename = f'history_{stage_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_filename)
    plt.close()

# --- Main Execution ---
def main():
    try:
        train_gen, val_gen, test_gen = create_data_generators()
    except Exception as e:
        print(f"Error initializing generators: {e}")
        return

    # --- Calculating Class Weights (CRITICAL STEP) ---
    print("\n--- Calculating Class Weights to handle imbalance ---")
    try:
        train_classes = train_gen.classes
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_classes),
            y=train_classes
        )
        class_weights_dict = dict(enumerate(class_weights))
        print("Class Weights applied:")
        for idx, weight in class_weights_dict.items():
            print(f"  {CLASSES[idx]}: {weight:.2f}")
    except Exception as e:
        print(f"Error computing class weights: {e}")
        class_weights_dict = None

    # 1. Build & Compile (Phase 1)
    model = build_model(num_classes=NUM_CLASSES)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = create_callbacks()

    # 2. Phase 1: Feature Extraction
    print("\n=== Phase 1: Training Top Layers (Feature Extraction) ===")
    history_initial = model.fit(
        train_gen,
        epochs=12, 
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights_dict # Applying weights
    )
    plot_training_history(history_initial, "Initial_Phase")

    # 3. Phase 2: Smart Fine-Tuning
    print("\n=== Phase 2: Smart Fine-Tuning (Layers 100+) ===")
    unfreeze_model_smartly(model)
    
    # Continue training
    initial_epoch_count = len(history_initial.epoch)
    
    history_fine_tune = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=initial_epoch_count,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights_dict # Applying weights
    )
    plot_training_history(history_fine_tune, "Fine_Tuning_Phase")

    # 4. Final Evaluation
    print("\n=== Phase 3: Final Test Set Evaluation ===")
    
    # Load best model
    try:
        best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("Loaded best model for testing.")
    except:
        best_model = model

    if test_gen:
        print("Running detailed evaluation...")
        test_loss, test_acc = best_model.evaluate(test_gen)
        print(f"Final Test Accuracy: {test_acc:.4f}")

        # Classification Report
        predictions = best_model.predict(test_gen)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        class_labels = list(test_gen.class_indices.keys())

        print("\n" + "="*60)
        print(classification_report(true_classes, pred_classes, target_names=class_labels))
        print("="*60)
        
        # Confusion Matrix
        cm = confusion_matrix(true_classes, pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_final.png')
        plt.close()
    else:
        print("Test data not found, skipping evaluation.")

    print(f"Training Done. Model: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()