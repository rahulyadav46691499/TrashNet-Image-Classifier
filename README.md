# ♻️ TrashNet Image Classification Project (PyTorch + Streamlit)

This repository documents my end-to-end learning journey in Deep Learning with PyTorch. It moves from building a Convolutional Neural Network (CNN) from scratch to fine-tuning a pretrained VGG16 model, and finally deploying the solution as an interactive web application using Streamlit.

**Goal:** To classify waste images into one of six categories to assist in waste management and sorting.

### 🏷️ Classes
* Glass
* Paper
* Cardboard
* Plastic
* Metal
* Trash

---

## 🔹 1. CNN From Scratch (Baseline Model)
### 📄 File: cnn_from_scratch.ipynb

In this step, I built a custom CNN architecture to understand the fundamentals of deep learning without relying on pre-built backbones.

#### What I did:
* Designed a custom architecture using nn.Conv2d, nn.ReLU, and nn.MaxPool2d.
* Implemented custom training and validation loops.
* Tracked accuracy and loss manually.

#### Key Learning Outcomes:
* Understanding convolution, pooling, and fully connected layers.
* Backpropagation and gradient updates.
* Visualizing Overfitting vs. Underfitting.
* The importance of data normalization.
* Result: ✅ Test Accuracy: ~84%

## 🔹 2. Fine-Tuning a Pretrained Model (VGG16)
### 📄 File: vgg16_fine_tuning.py

To improve performance and stability, I leveraged Transfer Learning using the VGG16 architecture pretrained on ImageNet.

#### What I did:
* Loaded VGG16 and froze the convolutional feature extractor layers.
* Replaced the final classifier head with a custom fully connected layer for 6 classes.
* Applied standard ImageNet normalization.

#### Key Learning Outcomes:
* Transfer learning concepts vs. training from scratch.
* Why pretrained models converge faster.
* Techniques for freezing and unfreezing layers.
* Result: ✅ Test Accuracy: ~88-90% (Faster convergence and more stable training).

### Output: Saved the best model as VGG16_model.pth.

## 🔹 3. Model Deployment with Streamlit
### 📄 File: app.py

The final step was bringing the model to life with a user-friendly interface.

#### What I did:
* Built a web app allowing users to upload .jpg, .jpeg, or .png images.
* Implemented cached model loading (@st.cache_resource) for performance.
* Added logic to handle inference on CPU/GPU automatically.

#### Key Learning Outcomes:
* Deployment basics and bridging the gap between Notebooks and Production.
* Handling inference-time transformations (ensuring they match training).
* Debugging real-world ML issues (e.g., class mapping mismatches).

## 🚀 How to Run the App

### Clone the repository:
```git clone <your-repo-url>```

```cd <your-repo-folder>```

### Install dependencies:
```pip install -r requirements.txt```

### Run the Streamlit app:
```streamlit run 03_streamlit_app.py```

Open in browser: Navigate to ```http://localhost:8501``` to test the classifier!

## 📊 Dataset Info

* Dataset Name: TrashNet
* Source: Public academic dataset for waste classification.
* Size: ~2,500+ images.
* Format: RGB images of waste on a white background.
