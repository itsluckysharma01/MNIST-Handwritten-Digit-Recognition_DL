# 🧠 Hand Digit Recognition using CNN

A deep learning web application that recognizes handwritten digits (0-9) using Convolutional Neural Networks (CNN). Built with TensorFlow/Keras and Streamlit, this app provides an interactive interface to either draw digits or upload images for real-time predictions.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red.svg)](https://streamlit.io/)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Training Your Own Model](#training-your-own-model)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## ✨ Features

- **🎨 Interactive Drawing Canvas**: Draw digits directly in the browser
- **📁 Image Upload**: Upload images of handwritten digits (PNG, JPG, JPEG, BMP)
- **🔄 Advanced Preprocessing**: Automatic image correction for better accuracy
  - Auto-inversion for dark-on-light or light-on-dark images
  - Noise reduction and morphological operations
  - Smart centering and aspect ratio preservation
- **📊 Confidence Scores**: View prediction confidence and probability distributions
- **🚀 Real-time Predictions**: Instant digit recognition
- **📈 Visual Feedback**: See original vs. preprocessed images side-by-side

## 🎥 Demo

### Drawing Interface

Users can draw digits on an interactive canvas and get instant predictions.

### Image Upload

Upload any handwritten digit image - the app automatically:

1. Converts to grayscale
2. Inverts if needed (MNIST expects white digits on black background)
3. Removes noise
4. Centers and resizes to 28×28 pixels
5. Makes prediction with confidence score

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Steps

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd Hand_Digit_Minist
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify installation**

```bash
python -m streamlit --version
```

## 💻 Usage

### Running the Web App

```bash
python -m streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Using the App

#### Option 1: Draw Mode

1. Select the **✏️ Draw Digit** tab
2. Draw a digit (0-9) on the canvas
3. Click **Predict**
4. View the prediction and confidence score

#### Option 2: Upload Mode

1. Select the **📁 Upload Image** tab
2. Click "Choose an image" and upload a digit image
3. View:
   - Original image
   - Preprocessed image (how the model sees it)
   - Prediction with confidence
   - Probability distribution for all digits

### Tips for Best Results

- Use clear, single-digit images
- Digits can be dark-on-light or light-on-dark
- Avoid cluttered backgrounds
- Center the digit in the image

## 🏗️ Model Architecture

### CNN Model (Improved)

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
dropout (Dropout)            (None, 13, 13, 32)        0
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18,496
max_pooling2d_1 (MaxPooling)  (None, 5, 5, 64)         0
dropout_1 (Dropout)          (None, 5, 5, 64)          0
flatten (Flatten)            (None, 1600)              0
dense (Dense)                (None, 128)               204,928
dropout_2 (Dropout)          (None, 128)               0
dense_1 (Dense)              (None, 10)                1,290
=================================================================
Total params: 225,034 (879.04 KB)
Trainable params: 225,034 (879.04 KB)
Non-trainable params: 0 (0.00 B)
```

### Key Features:

- **2 Convolutional Blocks**: Extract spatial features
- **Dropout Layers**: Prevent overfitting (25% and 50%)
- **Max Pooling**: Reduce dimensionality
- **Dense Layer**: Final classification

## 📊 Performance

| Metric              | Value                                |
| ------------------- | ------------------------------------ |
| **Test Accuracy**   | ~99.0%+                              |
| **Training Epochs** | 10                                   |
| **Dataset**         | MNIST (60,000 training, 10,000 test) |
| **Model Size**      | 879 KB                               |
| **Inference Time**  | <100ms                               |

### Training Progress

- Epoch 1: 98.10% validation accuracy
- Epoch 3: 99.03% validation accuracy
- Epoch 5+: 99.05%+ validation accuracy

## 📁 Project Structure

```
Hand_Digit_Minist/
│
├── app.py                      # Streamlit web application
├── train_improved_model.py     # Train the CNN model
├── digit.ipynb                 # Jupyter notebook (exploration)
├── mnist_model.h5              # Simple Dense model (backup)
├── mnist_cnn_model.h5          # CNN model (main)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── training_history.png        # Training visualization (generated)
```

## 🎓 Training Your Own Model

To train a new model from scratch:

```bash
python train_improved_model.py
```

This will:

1. Download the MNIST dataset
2. Build and compile the CNN model
3. Train for 10 epochs with validation
4. Save the model as `mnist_cnn_model.h5`
5. Generate training history visualization

### Customization

Edit `train_improved_model.py` to modify:

- **Epochs**: Change `epochs=10` to train longer/shorter
- **Batch size**: Adjust `batch_size=128`
- **Architecture**: Add/remove layers in the model definition
- **Dropout rates**: Modify dropout percentages

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select your repository
5. Deploy!

**Note**: Make sure to include `mnist_cnn_model.h5` in your repository.

### Alternative Platforms

- **Hugging Face Spaces**: [huggingface.co/spaces](https://huggingface.co/spaces)
- **Render**: [render.com](https://render.com)
- **Railway**: [railway.app](https://railway.app)

## 🛠️ Technologies Used

- **[TensorFlow/Keras](https://www.tensorflow.org/)**: Deep learning framework
- **[Streamlit](https://streamlit.io/)**: Web app framework
- **[OpenCV](https://opencv.org/)**: Image preprocessing
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[Pillow](https://python-pillow.org/)**: Image processing
- **[streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)**: Interactive drawing

## 🔮 Future Improvements

- [ ] Add data augmentation for better generalization
- [ ] Support for multi-digit recognition
- [ ] Export predictions to CSV
- [ ] Add model comparison (CNN vs Dense vs other architectures)
- [ ] Real-time video digit recognition
- [ ] Support for other handwriting datasets (EMNIST, etc.)
- [ ] Model quantization for faster inference
- [ ] Docker containerization
- [ ] API endpoint for programmatic access

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created with ❤️ using TensorFlow and Streamlit

## 🙏 Acknowledgments

- MNIST Dataset by Yann LeCun
- Streamlit Community
- TensorFlow/Keras Documentation

---

**⭐ If you find this project useful, please consider giving it a star!**
