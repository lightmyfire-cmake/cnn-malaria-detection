# 🦠 CNN Malaria Detection

A Streamlit-based web app using a Convolutional Neural Network (CNN) to detect malaria-infected cells from microscopic images.

---

## 📦 Features

- 🖼 Browse through cell images via an interactive UI
- 🔍 Select or randomly pick an image to analyze
- 🧠 Predict malaria infection using a trained CNN
- 🔥 View Saliency Maps to understand which parts of the image influenced the prediction
- 📊 Includes real vs predicted label comparison for evaluation

---

## 🛠️ Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [NumPy, PIL](https://pypi.org/)

---

## 🧪 Running the App Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/cnn-malaria-detection.git
cd cnn-malaria-detection
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
cnn-malaria-detection/
├── app.py                    # Entry point
├── core/
│   └── malaria_app.py        # Streamlit controller
├── model/
│   └── malaria_model.py      # CNN model logic & saliency map
├── ui/
│   └── image_browser.py      # Image selector & label browser
├── test_cell_images/         # Microscopic test images
├── tests/                    # Unit tests
├── requirements.txt
└── README.md
```

---

## ✅ Unit Tests

```bash
python -m unittest discover tests
```

---

## 📈 Model Performance

- Accuracy: **99%**
- Recall (Sensitivity): **99%**

---

## 📚 Data Source

NIH Malaria Dataset (27,558 images):  
🔗 [https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)

---

## 🧠 How It Works

The app loads a pre-trained CNN model (`.keras`) that classifies cell images as:

- **Infected**  
- **Uninfected**

Saliency maps visualize **which areas influenced** the model’s predictions, helping explain its behavior.