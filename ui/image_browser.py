import os
import random
from PIL import Image
import streamlit as st

class ImageBrowser:
    def __init__(self, base_folder="test_cell_images"):
        self.categories = {"parasitized": 1, "uninfected": 0}
        self.base_folder = base_folder
        self.images_per_page = 100
        self._load_images()

    def _load_images(self):
        if "shuffled_images" not in st.session_state:
            image_files = []
            labels = []
            for category, label in self.categories.items():
                folder_path = os.path.join(self.base_folder, category)
                for filename in os.listdir(folder_path):
                    full_path = os.path.normpath(os.path.join(folder_path, filename))
                    image_files.append(full_path)
                    labels.append(label)

            indices = list(range(len(image_files)))
            random.shuffle(indices)

            st.session_state.shuffled_images = [image_files[i] for i in indices]
            st.session_state.shuffled_labels = [labels[i] for i in indices]

        if "page_number" not in st.session_state:
            st.session_state.page_number = 1

        if "selected_image" not in st.session_state:
            st.session_state.selected_image = st.session_state.shuffled_images[0]

    def sidebar_image_selector(self):
        st.write("### Select an Image:")
        start = (st.session_state.page_number - 1) * self.images_per_page
        end = start + self.images_per_page
        cols = st.columns(5)

        for i, img_path in enumerate(st.session_state.shuffled_images[start:end]):
            img = Image.open(img_path).resize((100, 100))
            with cols[i % 5]:
                st.image(img, use_container_width=True)
                if st.button("Select", key=img_path):
                    st.session_state.selected_image = img_path
                    st.rerun()

    def get_selected_image(self):
        return st.session_state.selected_image

    def get_actual_label(self):
        path = os.path.normpath(st.session_state.selected_image)
        if path in st.session_state.shuffled_images:
            idx = st.session_state.shuffled_images.index(path)
            return "Infected" if st.session_state.shuffled_labels[idx] == 1 else "Healthy"
        return "Unknown"

    def random_image(self):
        st.session_state.selected_image = random.choice(st.session_state.shuffled_images)
        st.rerun()
