# Chest X-Ray Pneumonia Detection with PyTorch

This project uses deep learning with PyTorch to classify chest X-ray images as either **Pneumonia** or **Normal**. It supports multiple CNN architectures like AlexNet, DenseNet, EfficientNet, and ResNet.

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/MarwanAbdellah/Chest_X-Ray_Pneumonia_PyTorch.git
cd Chest_X-Ray_Pneumonia_PyTorch
```
2. **(Optional) Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```
## 🧠 Model Architectures
This repo includes these CNN models:

* AlexNet

* DenseNet

* EfficientNet

* ResNet

Each model is in its own folder. You can train and test them separately.

## 🗂️ Dataset
The dataset used is [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). It has two categories: PNEUMONIA and NORMAL. Make sure to download it from Kaggle and place it correctly in your project.

## 🚀 How to Use
1. Go into the model folder you want (like Resnet):

```bash
cd Resnet
```
2. Train the model:

```bash
python train.py
```
3. Evaluate it:

```bash
python evaluate.py
```
4. For visualization and experiments, check the Jupyter notebooks inside the notebooks folder.

## 📊 Results
You can add your accuracy, confusion matrix, or graphs here after training.

## 🤝 Contributing
Feel free to open issues or pull requests if you'd like to help improve the project.

## 📄 License
This project is licensed under the MIT License.

## 📧 Contact
Made with ❤️ by Marwan Abdellah
