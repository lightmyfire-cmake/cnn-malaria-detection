import tensorflow as tf
import numpy as np

class MalariaModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def preprocess_image(self, image, image_size=(64, 64)):
        image = image.resize(image_size)
        image_array = np.array(image) / 255.0
        return image_array

    def predict(self, image_array):
        return self.model.predict(np.array([image_array]))[0]

    def compute_saliency(self, img_array):
        x = tf.convert_to_tensor([img_array], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = self.model(x)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, x)
        saliency = tf.abs(grads)[0]
        saliency_2d = tf.reduce_max(saliency, axis=-1).numpy()

        saliency_2d -= saliency_2d.min()
        saliency_2d /= (saliency_2d.max() + 1e-8)
        return saliency_2d
