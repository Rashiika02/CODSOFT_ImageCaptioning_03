import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define the LSTM-based Captioning Model
class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out[:, -1])
        return outputs

# Load pre-trained ResNet model
def extract_features(image_path):
    model = models.resnet50(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image_tensor)
    return features

# Generate captions (dummy function, replace with actual model)
def generate_caption(features, model, vocab):
    # Dummy caption generation (replace with your trained model)
    return "a sample caption describing the image"

# Main function
def main():
    # Parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = 10000  # Adjust according to your vocabulary size
    image_path = 'example.jpg'
    
    # Load the pre-trained image model
    features = extract_features(image_path)
    
    # Initialize the captioning model
    model = CaptioningModel(embed_size, hidden_size, vocab_size)
    model.eval()
    
    # Generate a caption
    caption = generate_caption(features, model, vocab_size)
    
    print(f'Generated Caption: {caption}')

if __name__ == '__main__':
    main()
