import tensorflow as tf

# 1. Load the Keras model (using compile=False to avoid some errors)
model = tf.keras.models.load_model(r'C:\Users\winni\Downloads\best_wheat_model.keras')

# 2. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save the .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite successfully!")