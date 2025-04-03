import unittest
from PIL import Image
from model.malaria_model import MalariaModel
import numpy as np

class TestSaliency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MalariaModel("model.mdb_wts.keras")
        sample = Image.new("RGB", (64, 64), color="white")
        cls.img_array = cls.model.preprocess_image(sample)

    def test_saliency_output_shape_and_range(self):
        saliency = self.model.compute_saliency(self.img_array)
        self.assertEqual(saliency.shape, (64, 64))
        self.assertGreaterEqual(saliency.min(), 0.0)
        self.assertLessEqual(saliency.max(), 1.0)
