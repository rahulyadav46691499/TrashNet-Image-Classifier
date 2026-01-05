♻️ TrashNet Image Classification Project (PyTorch + Streamlit)

This repository documents my end-to-end learning journey in Deep Learning with PyTorch, moving from building a CNN from scratch to fine-tuning a pretrained model, and finally deploying the model using Streamlit.

The goal of the project is to classify trash images into one of the following categories:

Glass, Paper, Cardboard, Plastic, Metal, Trash

📂 Project Structure
.
├── 01_cnn_from_scratch.ipynb
├── 02_finetune_vgg16_trashnet.ipynb
├── 03_streamlit_app.py
├── VGG16_model.pth
├── requirements.txt
└── README.md

🧠 Learning Progression & Files Explanation
🔹 1. CNN From Scratch (Baseline Model)

📄 File: 01_cnn_from_scratch.ipynb

What I did:

Built a custom CNN architecture using PyTorch (nn.Conv2d, nn.ReLU, nn.MaxPool2d)

Trained the model from scratch on the TrashNet dataset

Implemented:

Custom training loop

Validation loop

Accuracy tracking

Key Learning Outcomes:

How convolution, pooling, and fully connected layers work

Backpropagation and gradient updates

Overfitting vs underfitting

Importance of data normalization

Result:

✅ Test Accuracy: ~84%

🔹 2. Fine-Tuning a Pretrained Model (VGG16)

📄 File: 02_finetune_vgg16_trashnet.ipynb

What I did:

Used VGG16 pretrained on ImageNet

Froze convolutional layers (feature extractor)

Replaced the classifier with a custom head

Fine-tuned on the TrashNet dataset

Used proper ImageNet normalization

Key Learning Outcomes:

Transfer learning vs training from scratch

Why pretrained models converge faster

How to freeze / unfreeze layers

Classifier design for fine-tuning

Result:

✅ Test Accuracy: ~85–87%

Faster convergence and more stable training

Saved Model:

VGG16_model.pth

🔹 3. Model Deployment with Streamlit

📄 File: 03_streamlit_app.py

What I did:

Built a user-friendly web app using Streamlit

Loaded the trained VGG16 model

Implemented real-time image inference

Displayed predicted class for uploaded images

Features:

Image upload (.jpg, .jpeg, .png)

GPU support (if available)

Cached model loading for fast inference

Clean UI layout

Key Learning Outcomes:

Model deployment basics

Inference vs training differences

Streamlit caching (@st.cache_resource)

Debugging real-world ML issues

🚀 How to Run the Streamlit App
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Make sure model file exists
VGG16_model.pth

3️⃣ Run the app
streamlit run 03_streamlit_app.py

4️⃣ Open in browser
http://localhost:8501

📊 Dataset

Dataset: TrashNet

Classes: 6

Images: ~2,500+

Source: Public academic dataset for waste classification

🛠 Tech Stack

Python

PyTorch

Torchvision

Streamlit

PIL

NumPy

📌 Key Takeaways

Built an image classifier from scratch

Improved performance using transfer learning

Learned fine-tuning best practices

Successfully deployed an ML model

Understood real-world challenges like:

Class order mismatch

Transform mismatch

Inference optimization
