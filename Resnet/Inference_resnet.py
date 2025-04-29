import torch
from torchvision import transforms, models
from torch import nn
import joblib
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model with pretrained ImageNet weights
resnet = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V2)

# Modify first layer to accept grayscale images (1 channel)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify the final fully connected layer for binary classification
resnet.fc = nn.Linear(in_features=2048, out_features=1, bias=True)

# Load the trained weights for the model
try:
    resnet.load_state_dict(torch.load("Resnet/models/resnet_binary_classification.pth"))
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

# Move the model to the correct device (GPU/CPU)
resnet = resnet.to(device)

# Load label encoder for decoding the prediction
try:
    label_encoder = joblib.load("Resnet/models/label_encoder.pkl")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    exit()

# Inference function
def inference(img_gray):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img_gray).unsqueeze(0).to(device)

    resnet.eval()
    with torch.no_grad():
        output = resnet(img_tensor)
        prob = torch.sigmoid(output).item()
        class_idx = int(prob > 0.5)

    final_pred = label_encoder.inverse_transform([class_idx])[0]
    return final_pred
