import tensorflow as tf

model = tf.keras.models.load_model("plant_model.h5")
model.save("plant_model.keras")