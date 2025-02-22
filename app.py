import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt


def weighted_binary_crossentropy(weight_for_fn = 3):
  def loss(y_true, y_pred):
    # standard binary crossentropy loss
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # mask which is supposed to catch false negatives
    # when actual value is 1, the closer predicted value to 0 - the more this mask weights
    fn_mask = y_true * (1 - y_pred)

    # additional custom weight to make false negatives cost more
    return bce + weight_for_fn * fn_mask
  return loss

# Load Keras Model
model = tf.keras.models.load_model("model.mdb_wts.keras", compile=False)


# Preprocessing Function (Matches Training Preprocessing)
def preprocess_image(image, image_size=(64, 64)):
    image = image.resize(image_size)
    image_array = np.array(image) / 255.0  # Normalize
    return image_array  # No batch dimension added


import tensorflow as tf
import numpy as np

def compute_saliency(model, img_array):
    """
    model: a tf.keras.Model that outputs a single sigmoid for binary classification
    img_array: preprocessed image of shape (H, W, 3)
    """
    # Ensure batch dimension
    x = tf.convert_to_tensor([img_array], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x)  # shape (1, 1) for binary
        # We'll use the prediction for the single output unit
        loss = predictions[:, 0]

    grads = tape.gradient(loss, x)  # shape (1, H, W, 3)
    saliency = tf.abs(grads)[0]     # remove the batch dimension => shape (H, W, 3)
    
    # Optionally reduce across color channels to get a 2D map
    saliency_2d = tf.reduce_max(saliency, axis=-1)
    saliency_2d = saliency_2d.numpy()

    # Normalize between [0, 1]
    saliency_2d -= saliency_2d.min()
    saliency_2d /= (saliency_2d.max() + 1e-8)
    return saliency_2d





# Load Images from Test Dataset
BASE_FOLDER = "test_cell_images"
CATEGORIES = {"parasitized": 1, "uninfected": 0}
image_files = []
labels = {}

if "shuffled_images" not in st.session_state:
    image_files = []
    labels = []

    # Load images and corresponding labels
    for category, label in CATEGORIES.items():
        folder_path = os.path.join(BASE_FOLDER, category)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            norm_path = os.path.normpath(image_path)  # Normalize path format
            image_files.append(norm_path)
            labels.append(label)  # Keep corresponding labels in the same order

    # Shuffle images and labels together
    combined = list(range(len(image_files)))  # Create index mapping
    random.shuffle(combined)  # Shuffle index mapping

    # Apply shuffled index order
    shuffled_image_files = [image_files[i] for i in combined]
    shuffled_labels = [labels[i] for i in combined]

    # Store in session state
    st.session_state.shuffled_images = shuffled_image_files
    st.session_state.shuffled_labels = shuffled_labels

# Retrieve the shuffled images and labels from session state
image_files = st.session_state.shuffled_images
labels = st.session_state.shuffled_labels

# Pagination Settings
IMAGES_PER_PAGE = 100
num_pages = (len(image_files) // IMAGES_PER_PAGE) + 1
if "page_number" not in st.session_state:
    st.session_state.page_number = 1

if "selected_image" not in st.session_state:
    st.session_state.selected_image = image_files[0]

with st.sidebar:

    st.write("### Select an Image:")
    # Get Images for Selected Page
    start_index = (st.session_state.page_number - 1) * IMAGES_PER_PAGE
    end_index = start_index + IMAGES_PER_PAGE
    # Display images as buttons with pagination
    cols = st.columns(5)  # Create 5 columns for image selection
    
    for index, image_path in enumerate(image_files[start_index:end_index]):
        img = Image.open(image_path).resize((100, 100))  # Resize for thumbnail display
        with cols[index % 5]:
            st.image(img, use_container_width=True)
            if st.button("Select", key=image_path):  # Button without text but clickable
                st.session_state.selected_image = image_path
                st.rerun()  # Force immediate update
    
    current_page = st.session_state.page_number

# Streamlit App
st.write("## 🦠 Malaria Cell Detection using AI")
st.write("""
         Neural Network analyzes microscopic cell images to detect malaria infections.

### 📊 Model Performance:
- **Accuracy**: 99% ✅
- **Recall (Sensitivity)**: 99% 🏆  
  *(High recall ensures that malaria-infected cells are rarely missed.)*

### 🔗 Data Source:
This model was trained on the **NIH Malaria Dataset**, a publicly available collection of **27,558 cell images**.
[Click here to view the dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html).
""")
st.write("""
### 🔬 How to Use This Tool:
1. **Select an image** from the sidebar or use the **random image button**.
2. Click **'Detect Malaria'** to let the AI analyze the cell.
3. View the **prediction results** and **Saliency Map** to see how the AI made its decision.
""")


# Button to select a random image
if st.button("🔀 Random Image"):
    st.session_state.selected_image = random.choice(st.session_state.shuffled_images)
    st.rerun()

image = Image.open(st.session_state.selected_image).resize((200, 200))
st.image(image, caption="Selected Cell Image", use_container_width=False)


# Predict
if st.button("Detect Malaria"): 
    img_array = preprocess_image(image)
    prediction = model.predict(np.array([img_array]))[0]  # Ensure correct shape
    infected_prob = prediction[0] * 100
    healthy_prob = 100 - infected_prob
    
    result = "Infected" if infected_prob > 50 else "Healthy"
    # Find index of the selected image
    selected_image_path = os.path.normpath(st.session_state.selected_image)

    if selected_image_path in st.session_state.shuffled_images:
        index = st.session_state.shuffled_images.index(selected_image_path)
        selected_label = st.session_state.shuffled_labels[index]  # Get the corresponding label
        real_label = "Infected" if selected_label == 1 else "Healthy"
    else:
        real_label = "Unknown"

    
    st.write(f"### Prediction: {result}")
    st.write(f"🩸 Infected Probability: {infected_prob:.2f}%")
    st.write(f"🩺 Healthy Probability: {healthy_prob:.2f}%")
    
    # Show real label
    st.write(f"### Actual Label: {real_label}")
    
    # Generate Grad-CAM
    st.write("## 🔍 Explainability with Saliency Map")
    st.write("""
    The Saliency Map highlights the most important areas of the cell that the AI used to make its prediction. 
    This helps understand **why the model thinks the cell is infected or healthy**.

    - **Bright areas** indicate regions that contributed **strongly** to the decision.
    - **Darker areas** mean those parts had **less influence** on the AI's prediction.

    This technique enhances **AI transparency**, making it easier to trust the model's decisions.
    """)

    saliency_map = compute_saliency(model, img_array)

    # Resize saliency map to match the image dimensions (64x64 in this case)
    saliency_map_resized = cv2.resize(saliency_map, (200, 200))

    # Normalize to [0,255] and apply colormap
    heatmap = cv2.applyColorMap((saliency_map_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    # Resize original image to match overlay dimensions
    original_resized = cv2.resize((img_array * 255).astype(np.uint8), (200, 200))

    # Overlay heatmap onto the original image
    overlay = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)

    # Display in Streamlit
    st.image(overlay, caption="Saliency Map Visualization", use_container_width=False)

    st.write("""
### 🧪 What Does This Mean?
The overlay on the cell image represents how the AI focused on specific cell regions.
- If the brightest areas align with infected regions, the model is making reliable predictions.
- If the saliency map is scattered, the AI might not be confident in its prediction.
  
Try selecting different images or using the random image button to compare how the model behaves on various samples.
""")


