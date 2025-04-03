import unittest
from PIL import Image
import numpy as np
from model.malaria_model import MalariaModel

class TestMalariaModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MalariaModel("model.mdb_wts.keras")
        cls.sample_image = Image.new("RGB", (64, 64), color="white")

    def test_model_loaded(self):
        self.assertIsNotNone(self.model.model)

    def test_preprocess_image_shape(self):
        processed = self.model.preprocess_image(self.sample_image)
        self.assertEqual(processed.shape, (64, 64, 3))
        self.assertTrue((processed <= 1.0).all() and (processed >= 0.0).all())

    def test_prediction_output(self):
        image_array = self.model.preprocess_image(self.sample_image)
        pred = self.model.predict(image_array)
        self.assertEqual(pred.shape, (1,))
        self.assertTrue(0.0 <= pred[0] <= 1.0)
