from flask import Flask, render_template, redirect, request
import torch
import torch.nn as nn
import numpy as np
import re
import pickle
import os

# Define the BiLSTM model class (same as in train_multi.py)
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=3, dropout=0.4):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                embedding_dim if i == 0 else hidden_dim * 2,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if i < num_layers - 1 else 0
            ) for i in range(num_layers)
        ])
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            ) for _ in range(8)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 8, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
    def attention_net(self, lstm_output):
        attention_outputs = []
        for attention_head in self.attention_heads:
            attention_weights = attention_head(lstm_output)
            context_vector = attention_weights * lstm_output
            attention_outputs.append(torch.sum(context_vector, dim=1))
        return torch.cat(attention_outputs, dim=1)
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)
        lstm_out = embedded
        for i, lstm_layer in enumerate(self.lstm_layers):
            lstm_out, _ = lstm_layer(lstm_out)
            if i > 0:
                lstm_out = lstm_out + lstm_out
        attn_out = self.attention_net(lstm_out)
        return self.classifier(attn_out)

# Load the model and vocabulary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define label mapping directly
label_mapping = {
    0: "Negative (sad, depressed, exhausted)",
    1: "Positive (happy, excited, optimistic)",
    2: "Affectionate/Caring",
    3: "Angry/Frustrated",
    4: "Anxious/Fearful",
    5: "Surprised/Shocked"
}
print("Using label mapping:", label_mapping)

# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load vocabulary first to get vocab_size
try:
    # Load vocabulary first to get vocab_size
    vocab_path = os.path.join(current_dir, 'vocab.pkl')
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab_list = pickle.load(f)
        word2idx = {word: idx for idx, word in enumerate(vocab_list)}
        vocab_size = 4224  # Use the same vocab size as the trained model
        print(f"Using vocabulary size: {vocab_size}")

    # Initialize the model with the same architecture as training
    model = BiLSTM(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_classes=6,
        num_layers=3,
        dropout=0.4
    ).to(device)

    # Load the trained model weights
    model_path = os.path.join(current_dir, 'best_model.pth')
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode

    # Print model architecture for verification
    print("\nModel Architecture:")
    print(model)
    print("\nModel loaded successfully!")

except Exception as e:
    print(f"Error during model loading: {str(e)}")
    raise

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [token for token in text.split() if token.strip()]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    pred_result = ""
    quote = ""
    
    if request.method == 'POST':
        if 'input_data' in request.form:
            input_data = request.form['input_data']
            try:
                print("\n=== New Prediction ===")
                print("Input text:", input_data)
                
                # Process the input data
                words = simple_tokenize(input_data)
                print("Tokenized words:", words)
                
                # Handle unknown words more gracefully
                indices = []
                for word in words:
                    if word in word2idx:
                        indices.append(word2idx[word])
                    else:
                        indices.append(0)  # Use 0 for unknown words
                print("Word indices (first 10):", indices[:10])
                
                # Pad or truncate to max_length=150
                if len(indices) < 150:
                    indices = indices + [0] * (150 - len(indices))  # Use 0 for padding
                else:
                    indices = indices[:150]
                
                # Convert to tensor and make prediction
                input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
                print("Input tensor shape:", input_tensor.shape)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    print("Raw model output shape:", output.shape)
                    
                    # Get probabilities for each class
                    probabilities = torch.softmax(output, dim=1)[0]
                    print("\nProbabilities for each class:")
                    for i, prob in enumerate(probabilities.tolist()):
                        print(f"Class {i}: {prob:.4f}")
                    
                    prediction = torch.argmax(output, dim=1).item()
                    print(f"Predicted class: {prediction}")
                
                # Map prediction to emotion using direct mapping
                pred_result = label_mapping[prediction]
                print("Final prediction:", pred_result)
                
                # Set quote based on prediction result
                quotes = {
                    'Negative': "Every day may not be good, but there's something good in every day.",
                    'Positive': "Happiness is not something ready-made. It comes from your own actions.",
                    'Affectionate': "The best and most beautiful things in the world cannot be seen or even touched - they must be felt with the heart.",
                    'Angry': "Take a deep breath. It's just a bad day, not a bad life.",
                    'Anxious': "You are braver than you believe, stronger than you seem, and smarter than you think.",
                    'Surprised': "Life is full of surprises and miracles."
                }
                # Get the first word of the emotion as the key
                emotion_key = pred_result.split()[0]
                quote = quotes.get(emotion_key, "Every moment is a fresh beginning.")
                
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                pred_result = f"Prediction Error: {str(e)}"

    return render_template('index.html', prediction=pred_result, quote=quote)

if __name__ == "__main__":
    app.run(debug=True)