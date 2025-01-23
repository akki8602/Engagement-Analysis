from hsemotion.facial_emotions import HSEmotionRecognizer
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import cv2 as cv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def image_conversion (path):
    face_img = Image.open(path)
    face_img = np.array(face_img)
    if face_img.max() <= 1:
        face_img = (face_img * 255).astype(np.uint8)
    if face_img.shape[-1] == 4:  # Check if the image has 4 channels
        face_img = Image.fromarray(face_img).convert("RGB")
    else:
        face_img = Image.fromarray(face_img)
    return np.array(face_img)

model_name = 'enet_b0_8_best_afew'
fer=HSEmotionRecognizer(model_name=model_name,device='cpu') # device is cpu or gpu
images = []

for image_path in ["Emotions_Test/Smile.png", "Emotions_Test/Neutral.png", "Emotions_Test/Shocked.png", "Emotions_Test/Bored.png", "Emotions_Test/Bored:Annoyed.png", "Emotions_Test/WT*.png"]:
    face_img = image_conversion(image_path)
    emotion,scores=fer.predict_emotions(face_img,logits=True)
    print(emotion)

torch.manual_seed(42)
np.random.seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize to model input size
    transforms.RandomHorizontalFlip(),           # Horizontal flip
    transforms.RandomRotation(10),              # Rotate ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust colors
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

dataset_path = "Finetuning_Dataset/"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

from torch.utils.data import DataLoader, WeightedRandomSampler

# Compute class weights
class_counts = [48, 35]  # Number of samples in each class
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)


train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, test_dataset = random_split(dataset, [train_size, val_size])

print("length of dataset: ",len(dataset))
print("length of train_dataset: ", len(train_dataset))
print("length of test_dataset: ", len(test_dataset))

sample_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=train_size, replacement=True)
train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
eval_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
print("length of train_loader: ", len(train_loader))
print("length of eval_loader: ", len(eval_loader))

model = fer.model
# Freeze the feature extractor
for param in model.parameters():
    param.requires_grad = False
for param in model.blocks[-2].parameters():  # Unfreeze the last block
    param.requires_grad = True

print(type(model.classifier))

dummy_input = torch.randn(8, 3, 224, 224)  # Adjust size based on your input dimensions
with torch.no_grad():
    features = model.forward_features(dummy_input)  # Most timm models have this method
    in_features = features.shape[1] 
model.classifier = nn.Linear(in_features, 2)
model.eval()  # Set model to evaluation mode
correct, total = 0, 0

with torch.no_grad():  # Disable gradient computation for validation
    for images, labels in eval_loader:
        images, labels = images.to("cpu"), labels.to("cpu")  # Adjust for GPU if needed
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy prior to training: {100 * correct / total:.2f}%")

optimizer = optim.Adam(model.classifier.parameters(), lr=0.008)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
weights = torch.tensor([1.0 / 48, 1.0 / 35], dtype=torch.float).to("cpu")
criterion = nn.CrossEntropyLoss(weight=weights)

for epoch in range(15):  # Example: 10 epochs
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to("cpu"), labels.to("cpu")  # Adjust to "cuda" if using GPU

        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update classifier weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    scheduler.step()
                      
model.eval()  # Set model to evaluation mode
correct, total = 0, 0
threshold = 0.47


y_true = []
y_pred = []
total = 0
correct = 0

with torch.no_grad():  # Disable gradient computation for validation
    for images, labels in eval_loader:
        images, labels = images.to("cpu"), labels.to("cpu")  # Adjust for GPU if needed
        outputs = model(images)  # Forward pass
        # _, preds = torch.max(outputs, 1)  # Get predicted class
        # total += labels.size(0)
        # correct += (preds == labels).sum().item()
        # probabilities = torch.softmax(outputs, dim = 1)
        # predicted = (probabilities[:,1] > threshold).int()
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

print(f"Eval Accuracy: {100 * correct / total:.2f}%")
accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")
print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"]))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Confused", "Normal"], yticklabels=["Confused", "Normal"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


torch.save(model, "enet_b0_8_best_afew_binary_finetuned_full.pth")

