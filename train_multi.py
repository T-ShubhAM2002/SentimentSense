import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import re
from sklearn.model_selection import train_test_split
import pickle
import torch.nn.functional as F

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
df = pd.read_csv('test.csv')
print(f"Dataset size: {len(df)}")

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Prepare data
X_train = train_df['text']
X_test = test_df['text']
y_train = train_df['label']
y_test = test_df['label']

def simple_tokenize(text):
    # Convert to lowercase and split on whitespace and punctuation
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace and filter empty strings
    return [token for token in text.split() if token.strip()]

# Build vocabulary
def build_vocab(texts, max_vocab_size=15000):  # Increased vocabulary size
    word_counts = Counter()
    for text in texts:
        words = simple_tokenize(text)
        word_counts.update(words)
    
    # Create vocabulary
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(max_vocab_size-2)]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx, vocab

# Build vocabulary from training data
word2idx, vocab = build_vocab(X_train)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Compute class weights with stronger emphasis on minority classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
# Increase weights for minority classes more aggressively
class_weights = class_weights * 2.0  # Doubled the weights
class_weights = torch.FloatTensor(class_weights).to(device)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train_onehot = torch.tensor(pd.get_dummies(y_train).values, dtype=torch.float32)
y_test_onehot = torch.tensor(pd.get_dummies(y_test).values, dtype=torch.float32)

# Create custom dataset with enhanced data augmentation
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_length=150, augment=False):  # Increased max_length
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def augment_text(self, text):
        words = simple_tokenize(text)
        if len(words) > 3:
            # Multiple augmentation techniques
            augmented = words.copy()
            
            # Random word dropout
            if np.random.random() < 0.3:  # 30% chance
                drop_prob = 0.1
                augmented = [w for w in augmented if np.random.random() > drop_prob]
            
            # Random word swap
            if np.random.random() < 0.3 and len(augmented) > 1:  # 30% chance
                idx1, idx2 = np.random.choice(len(augmented), 2, replace=False)
                augmented[idx1], augmented[idx2] = augmented[idx2], augmented[idx1]
            
            return ' '.join(augmented)
        return ' '.join(words)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        
        if self.augment:
            text = self.augment_text(text)
            
        words = simple_tokenize(text)
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        if len(indices) < self.max_length:
            indices = indices + [self.word2idx['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': self.labels[idx]
        }

# Create data loaders with augmentation for training
train_dataset = TextDataset(X_train, y_train_onehot, word2idx, augment=True)
test_dataset = TextDataset(X_test, y_test_onehot, word2idx, augment=False)

# Use smaller batch size for better gradient updates
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced batch size
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the enhanced BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=3, dropout=0.4):
        super(BiLSTM, self).__init__()
        
        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # BiLSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                embedding_dim if i == 0 else hidden_dim * 2,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if i < num_layers - 1 else 0
            ) for i in range(num_layers)
        ])
        
        # Multi-head attention with more heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            ) for _ in range(8)  # Increased to 8 attention heads
        ])
        
        # Enhanced classification head with residual connections
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
        # Get embeddings
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM layers with residual connections
        lstm_out = embedded
        for i, lstm_layer in enumerate(self.lstm_layers):
            lstm_out, _ = lstm_layer(lstm_out)
            if i > 0:  # Add residual connection
                lstm_out = lstm_out + lstm_out
        
        # Apply multi-head attention
        attn_out = self.attention_net(lstm_out)
        
        # Classification with residual connection
        logits = self.classifier(attn_out)
        
        return logits

# Initialize model with adjusted parameters
embedding_dim = 300
hidden_dim = 256
model = BiLSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    num_layers=3,    # Ensure num_layers > 1 for dropout
    dropout=0.3      # Reduced dropout for better stability
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Calculate total steps for OneCycleLR
total_steps = len(train_loader) * 50  # Reduced epochs to 50
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1000.0
)

# Training function with gradient clipping
def train_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, true_labels = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == true_labels).sum().item()
    
    return total_loss / len(loader), correct / total

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
    
    return total_loss / len(loader), all_preds, all_labels

# Training loop with early stopping
print("Starting training...")
best_val_loss = float('inf')
patience = 5
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# Save vocabulary and word2idx for inference
with open('flask/vocab.pkl', 'wb') as f:
    pickle.dump({'vocab': vocab, 'word2idx': word2idx}, f)

# Save label mapping
label_mapping = {i: label for i, label in enumerate(sorted(np.unique(y_train)))}
with open('flask/label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

for epoch in range(50):  # Reduced epochs to 50
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
    
    # Validation
    val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
    val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'Epoch {epoch+1}/50:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'flask/best_model.pth')
        print("Saved new best model!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Load best model for final evaluation
model.load_state_dict(torch.load('flask/best_model.pth'))

# Final evaluation
print("\nFinal evaluation...")
_, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

# Print classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds))

# Create confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('flask/confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('flask/training_history.png')
plt.close()

print("\nTraining completed! Model and evaluation plots saved to flask directory.")

# After training loop, add:
# Save the model
torch.save(model.state_dict(), 'model.pth')

# Save the vocabulary
with open('vocab.pkl', 'wb') as f:
    pickle.dump(word2idx, f) 