import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

transform = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize to model input size
    transforms.RandomHorizontalFlip(),           # Horizontal flip
    transforms.RandomRotation(10),              # Rotate ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust colors
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

dataset_path = "Test_Dataset/"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = torch.load("enet_b0_8_best_afew_binary_finetuned_full.pth")
model.eval()  # Set model to evaluation mode
correct, total = 0, 0
threshold = 0.54
y_true = []
y_pred = []

with torch.no_grad():  
    for images, labels in test_loader:
        images, labels = images.to("cpu"), labels.to("cpu")  
        outputs = model(images)  
        probabilities = torch.softmax(outputs, dim=1)

        predicted = (probabilities[:, 0] > threshold).int()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        y_true.extend(labels.cpu().numpy())  # Ground truth labels
        y_pred.extend(predicted.cpu().numpy())  # Pr

print(f"Testing Accuracy: {100 * correct / total:.2f}%")
print(classification_report(y_true, y_pred, target_names=["Normal", "Confused"]))

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Confused", "Normal"], yticklabels=["Confused", "Normal"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



