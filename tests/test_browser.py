import unittest
from ui.image_browser import ImageBrowser
import streamlit as st

class TestImageBrowser(unittest.TestCase):
    def test_browser_initialization(self):
        browser = ImageBrowser()
        self.assertIn("shuffled_images", st.session_state)
        self.assertIn("shuffled_labels", st.session_state)
        self.assertEqual(len(st.session_state.shuffled_images), len(st.session_state.shuffled_labels))

    def test_selected_image_exists(self):
        browser = ImageBrowser()
        selected = browser.get_selected_image()
        self.assertTrue(any(selected.endswith(ext) for ext in [".jpg", ".png", ".jpeg"]))
