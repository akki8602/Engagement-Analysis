from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from hsemotion.facial_emotions import HSEmotionRecognizer
from flask_cors import CORS
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your fine-tuned model
model_path = "enet_b0_8_best_afew_binary_finetuned_full.pth"
model = torch.load(model_path)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Endpoint for health check
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Server is running!"})

# Endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    # Get the uploaded file
    file = request.files["file"]
    try:
        # Process the image
        image = Image.open(file).convert("RGB")
        orig_image = image.copy()
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make a prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            threshold = 0.47
            predicted = (probabilities[:, 1] > threshold).int()
        
        # Convert prediction to label (assuming binary classification)
        label_map = {0: "User is confused", 1: "User is normal"}
        label = label_map[predicted.item()]

        plt.switch_backend('agg')
        plt.figure()
        plt.imshow(orig_image)
        plt.axis('off')
        plt.savefig('capture.png')
        plt.close()
        
        return jsonify({"Engagement analysis prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
