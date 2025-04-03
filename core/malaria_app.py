import streamlit as st
from PIL import Image
import cv2
import numpy as np
from model.malaria_model import MalariaModel
from ui.image_browser import ImageBrowser

class MalariaApp:
    def __init__(self):
        self.model = MalariaModel("model.mdb_wts.keras")
        self.browser = ImageBrowser()

    def run(self):
        with st.sidebar:
            self.browser.sidebar_image_selector()

        st.title("🦠 Malaria Cell Detection using AI")
        self.render_intro()

        if st.button("🔀 Random Image"):
            self.browser.random_image()

        image_path = self.browser.get_selected_image()
        image = Image.open(image_path).resize((200, 200))
        st.image(image, caption="Selected Cell Image", use_container_width=False)

        if st.button("Detect Malaria"):
            img_array = self.model.preprocess_image(image)
            prediction = self.model.predict(img_array)
            infected_prob = prediction[0] * 100
            healthy_prob = 100 - infected_prob
            result = "Infected" if infected_prob > 50 else "Healthy"
            actual_label = self.browser.get_actual_label()

            st.markdown(f"### Prediction: **{result}**")
            st.write(f"🩸 Infected Probability: {infected_prob:.2f}%")
            st.write(f"🩺 Healthy Probability: {healthy_prob:.2f}%")
            st.write(f"### Actual Label: {actual_label}")

            self.render_saliency(img_array)

    def render_intro(self):
        st.markdown("""
        Neural Network analyzes microscopic cell images to detect malaria infections.

        ### 📊 Model Performance:
        - **Accuracy**: 99%
        - **Recall (Sensitivity)**: 99%

        ### 🔗 Data Source:
        Trained on the [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)
        """)

    def render_saliency(self, img_array):
        st.subheader("🔍 Explainability with Saliency Map")
        saliency_map = self.model.compute_saliency(img_array)
        saliency_resized = cv2.resize(saliency_map, (200, 200))
        heatmap = cv2.applyColorMap((saliency_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        original = cv2.resize((img_array * 255).astype(np.uint8), (200, 200))
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        st.image(overlay, caption="Saliency Map Visualization", use_container_width=False)

        st.markdown("""
        The saliency map shows areas that influenced the AI's decision:
        - **Bright areas** = high influence
        - **Dark areas** = low influence
        """)
