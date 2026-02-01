print("Converting to FULL INT8 TensorFlow Lite format for Raspberry Pi...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set representative dataset
converter.representative_dataset = representative_data_gen

# Force full INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set input/output type for Raspberry Pi
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8_model = converter.convert()

with open("resnet50_fpga_int8_rpi.tflite", "wb") as f:
    f.write(tflite_int8_model)

print("✓ Full INT8 TFLite model saved as 'resnet50_fpga_int8_rpi.tflite'")
