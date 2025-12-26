import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Success! GPU Detected: {gpus}")
else:
    print("❌ Error: No GPU found. Still running on CPU.")