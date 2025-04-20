from flask import Flask, request, jsonify
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import torch

app = Flask(__name__)

# Load model from .safetensors
model_path = "C:\\Users\\HP\\Downloads\\model\\my_emotion_classifier"  # Path to your folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move to GPU if available
device = 0 if torch.cuda.is_available() else -1
model = model.to(torch.device("cuda" if device == 0 else "cpu"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json.get('text', '')
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(model.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Apply sigmoid and threshold (0.6)
        probs = torch.sigmoid(outputs.logits)
        predictions = (probs >= 0.45).int().squeeze().tolist()
        
        # Map to label names
        id2label = model.config.id2label
        results = {
            "text": text,
            "predictions": {id2label[i]: int(pred) for i, pred in enumerate(predictions)},
            "probabilities": {id2label[i]: float(prob) for i, prob in enumerate(probs.squeeze().tolist())}
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)