import torch
from torchvision import transforms, models
from torch import nn
import joblib
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model with pretrained ImageNet weights
alexnet = models.alexnet(weights= models.AlexNet_Weights.IMAGENET1K_V1)

alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

# Load the trained weights for the model
try:
    alexnet.load_state_dict(torch.load("Alexnet/models/alexnet_binary_classification.pth"))
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

# Move the model to the correct device (GPU/CPU)
alexnet = alexnet.to(device)

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

    alexnet.eval()
    with torch.no_grad():
        output = alexnet(img_tensor) 
        class_idx = torch.argmax(output, dim=1).item()  

    final_pred = label_encoder.inverse_transform([class_idx])[0]
    return final_pred
