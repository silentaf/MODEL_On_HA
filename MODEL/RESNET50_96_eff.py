import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download("puneet6060/intel-image-classification")
print("Path to dataset files:", path)

# Dataset paths
data_dir = Path(path) / "seg_train" / "seg_train"
test_dir = Path(path) / "seg_test" / "seg_test"

# Configuration
IMG_SIZE = 224  # Increased from 150
BATCH_SIZE = 32
EPOCHS = 50

# Enhanced data augmentation pipeline
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nClasses: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Build Enhanced CNN with Residual Blocks
def residual_block(x, filters, kernel_size=3):
    """Residual block with skip connection"""
    shortcut = x
    
    # First conv layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second conv layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Match dimensions for skip connection if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# Build model with functional API
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Initial conv layer
x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

# Residual blocks - Stage 1
x = residual_block(x, 64)
x = residual_block(x, 64)
x = layers.MaxPooling2D(2)(x)

# Residual blocks - Stage 2
x = residual_block(x, 128)
x = residual_block(x, 128)
x = layers.MaxPooling2D(2)(x)

# Residual blocks - Stage 3
x = residual_block(x, 256)
x = residual_block(x, 256)
x = residual_block(x, 256)
x = layers.MaxPooling2D(2)(x)

# Residual blocks - Stage 4
x = residual_block(x, 512)
x = residual_block(x, 512)

# Global average pooling
x = layers.GlobalAveragePooling2D()(x)

# Dense layers with regularization
x = layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.3)(x)

# Output layer
outputs = layers.Dense(6, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile with optimized settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
)

print("\n" + "="*60)
print("ENHANCED MODEL ARCHITECTURE WITH RESIDUAL CONNECTIONS")
print("="*60)
model.summary()

# Advanced callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("\n" + "="*60)
print("TRAINING ENHANCED MODEL")
print("="*60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Load best model
model = keras.models.load_model('best_model.h5')

# Evaluate on test set
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

test_results = model.evaluate(test_generator, verbose=1)
test_loss = test_results[0]
test_accuracy = test_results[1]
test_top2_accuracy = test_results[2]

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Top-2 Accuracy: {test_top2_accuracy:.4f} ({test_top2_accuracy*100:.2f}%)")

# Get predictions
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(train_generator.class_indices.keys())

# Classification report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4))

# Per-class accuracy
cm = confusion_matrix(true_classes, predicted_classes)
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("\nPer-Class Accuracy:")
for i, label in enumerate(class_labels):
    print(f"  {label:12s}: {per_class_accuracy[i]:.4f} ({per_class_accuracy[i]*100:.2f}%)")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy plot
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss plot
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Top-2 Accuracy
axes[1, 0].plot(history.history['top_2_accuracy'], label='Train Top-2', linewidth=2)
axes[1, 0].plot(history.history['val_top_2_accuracy'], label='Val Top-2', linewidth=2)
axes[1, 0].set_title('Top-2 Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Per-class accuracy bar chart
axes[1, 1].bar(range(len(class_labels)), per_class_accuracy * 100, color='steelblue')
axes[1, 1].set_xticks(range(len(class_labels)))
axes[1, 1].set_xticklabels(class_labels, rotation=45, ha='right')
axes[1, 1].set_title('Per-Class Test Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('enhanced_training_metrics.png', dpi=300, bbox_inches='tight')
print("\n✓ Enhanced metrics plot saved as 'enhanced_training_metrics.png'")

# Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Enhanced Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'enhanced_confusion_matrix.png'")

# Save final model
model.save('intel_image_classifier_enhanced.h5')
print("\n✓ Enhanced model saved as 'intel_image_classifier_enhanced.h5'")

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"Architecture: ResNet-style with {len([l for l in model.layers if 'conv' in l.name])} convolutional layers")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f} ({max(history.history['val_accuracy'])*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Top-2 Accuracy: {test_top2_accuracy:.4f} ({test_top2_accuracy*100:.2f}%)")
print(f"Total Parameters: {model.count_params():,}")
print(f"Average Per-Class Accuracy: {per_class_accuracy.mean():.4f} ({per_class_accuracy.mean()*100:.2f}%)")
print("="*60)
