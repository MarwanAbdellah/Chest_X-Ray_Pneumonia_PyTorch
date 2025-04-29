import torch
from torchvision import transforms, models
from torch import nn
import joblib
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model with pretrained ImageNet weights
densenet = models.densenet201(weights= models.DenseNet201_Weights.IMAGENET1K_V1)

# Modify first layer to accept grayscale images (1 channel)
densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

densenet.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)

# Load the trained weights for the model
try:
    densenet.load_state_dict(torch.load("Densenet/models/densenet_binary_classification.pth"))
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

# Move the model to the correct device (GPU/CPU)
densenet = densenet.to(device)

# Load label encoder for decoding the prediction
try:
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    exit()

def inference(img_gray):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img_gray).unsqueeze(0).to(device)

    densenet.eval()
    with torch.no_grad():
        output = densenet(img_tensor)  # shape: [1, num_classes]
        class_idx = torch.argmax(output, dim=1).item()  # get index of the highest score

    final_pred = label_encoder.inverse_transform([class_idx])[0]
    return final_pred