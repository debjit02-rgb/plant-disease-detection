import os
os.environ["OMP_NUM_THREADS"] = "1"

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("plant_model.keras")

class_names = sorted(os.listdir("dataset"))

img_path = "test.jpg"

img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
class_index = np.argmax(predictions)
confidence = np.max(predictions)

print("Prediction:", class_names[class_index])
print("Confidence:", round(confidence * 100, 2), "%")