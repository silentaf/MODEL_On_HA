import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import kagglehub

# Download dataset
print("="*70)
print("DOWNLOADING INTEL IMAGE CLASSIFICATION DATASET")
print("="*70)
path = kagglehub.dataset_download("puneet6060/intel-image-classification")
print(f"Path to dataset files: {path}\n")

# Simplified ResNet Block - FPGA friendly
class BasicBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride

        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        # Shortcut
        self.shortcut_conv = None
        self.shortcut_bn = None

        self.relu2 = layers.ReLU()

    def build(self, input_shape):
        # Create shortcut if dimensions change
        if self.stride != 1 or input_shape[-1] != self.filters:
            self.shortcut_conv = layers.Conv2D(self.filters, 1, strides=self.stride,
                                               padding='same', use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()

    def call(self, x, training=False):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        # Apply shortcut
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)

        out = layers.add([out, shortcut])
        out = self.relu2(out)

        return out

# Build Simplified ResNet-50 (FPGA friendly)
def build_resnet50_fpga(input_shape=(150, 150, 3), num_classes=6):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Layer 1: 2 blocks, 64 filters
    for _ in range(2):
        x = BasicBlock(64, stride=1)(x)

    # Layer 2: 3 blocks, 128 filters
    x = BasicBlock(128, stride=2)(x)
    for _ in range(2):
        x = BasicBlock(128, stride=1)(x)

    # Layer 3: 4 blocks, 256 filters
    x = BasicBlock(256, stride=2)(x)
    for _ in range(3):
        x = BasicBlock(256, stride=1)(x)

    # Layer 4: 2 blocks, 512 filters
    x = BasicBlock(512, stride=2)(x)
    for _ in range(1):
        x = BasicBlock(512, stride=1)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet50_FPGA')
    return model

# Plot training history
def plot_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history.history['loss'], label='Train Loss', marker='o')
    ax1.plot(history.history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved as '{save_path}'")
    plt.close()

# Custom callback for detailed logging
class DetailedLogging(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        print("\n" + "-"*70)
        print(f"EPOCH [{epoch+1}/{self.params['epochs']}]")
        print("-"*70)
        print(f"Train Loss:     {logs['loss']:.4f} | Train Acc: {logs['accuracy']*100:.2f}%")
        print(f"Val Loss:       {logs['val_loss']:.4f} | Val Acc:   {logs['val_accuracy']*100:.2f}%")

        # Check if this is the best model
        if logs['val_accuracy'] > self.best_val_acc:
            self.best_val_acc = logs['val_accuracy']
            print("✓ NEW BEST MODEL!")

        print("="*70)

# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("INTEL IMAGE CLASSIFICATION - ResNet50 (FPGA Optimized)")
    print("="*70 + "\n")

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("Running on CPU")
    print()

    # Dataset paths
    train_dir = os.path.join(path, 'seg_train/seg_train')
    test_dir = os.path.join(path, 'seg_test/seg_test')

    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.2  # Use 20% for validation
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Image size and batch size
    img_size = (150, 150)
    batch_size = 32

    print("Loading datasets...")

    # Training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation data
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Test data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print(f"\nDataset Statistics:")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Test samples: {test_generator.samples}")
    print(f"  Classes: {list(train_generator.class_indices.keys())}")
    print(f"  Number of classes: {train_generator.num_classes}")
    print()

    # Build model
    print("Creating model...")
    model = build_resnet50_fpga(input_shape=(150, 150, 3),
                                 num_classes=train_generator.num_classes)

    # Model summary
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model.summary()

    # Calculate model size
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size (float32): {total_params * 4 / 1e6:.2f} MB")
    print("="*70 + "\n")

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        DetailedLogging(),
        keras.callbacks.ModelCheckpoint(
            'best_resnet50_fpga.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            mode='max',
            verbose=1,
            min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            mode='max',
            verbose=1,
            restore_best_weights=True
        )
    ]

    # Train model
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70 + "\n")

    epochs = 25
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("="*70)

    # Plot training history
    plot_history(history)

    # Save final model
    model.save('final_resnet50_fpga.h5')
    model.save('final_resnet50_fpga.keras')  # New Keras format

    # Also save weights only (useful for deployment)
    model.save_weights('resnet50_fpga_weights.h5')

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"Final Train Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("="*70)

    print("\n" + "="*70)
    print("MODEL SAVED")
    print("="*70)
    print("Files saved:")
    print("  1. best_resnet50_fpga.h5 - Best model (full model)")
    print("  2. final_resnet50_fpga.h5 - Final model (full model)")
    print("  3. final_resnet50_fpga.keras - Final model (new format)")
    print("  4. resnet50_fpga_weights.h5 - Model weights only")
    print("  5. training_history.png - Training curves")
    print("="*70)

    print("\n" + "="*70)
    print("FPGA DEPLOYMENT NOTES")
    print("="*70)
    print("1. Model uses only basic operations (Conv2D, ReLU, BatchNorm)")
    print("2. No complex operations - fully synthesizable for FPGA")
    print("3. BatchNorm can be fused with Conv layers during deployment")
    print("4. Recommended: Apply INT8 quantization for FPGA optimization")
    print("5. Input size: 150x150x3, Output: 6 classes")
    print("6. Consider using TensorFlow Lite for model optimization")
    print("7. Use Xilinx Vitis AI or Intel FPGA AI Suite for deployment")
    print("="*70 + "\n")

    # Optional: Convert to TensorFlow Lite
    print("Converting to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('resnet50_fpga.tflite', 'wb') as f:
        f.write(tflite_model)
    print("✓ TensorFlow Lite model saved as 'resnet50_fpga.tflite'")
    print("  This optimized model is ready for edge deployment!\n")
