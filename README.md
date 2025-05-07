# SentimentSense - Mental Health Text Classification

SentimentSense is a deep learning project that classifies text into different mental health categories using a Bidirectional LSTM (BiLSTM) model with multi-head attention mechanism. The model is designed to analyze text input and classify it into various mental health categories, providing insights into the emotional and psychological state expressed in the text.

## Features

- Text classification into multiple mental health categories
- Real-time prediction through a web interface
- Advanced deep learning architecture with BiLSTM and attention
- Robust data preprocessing and augmentation
- Class imbalance handling
- Comprehensive performance metrics

## Project Structure

```
SentimentSense/
├── flask/                  # Flask web application
│   ├── app.py             # Main Flask application
│   ├── run.py             # Flask runner
│   ├── requirements.txt   # Flask dependencies
│   ├── templates/         # HTML templates
│   ├── static/           # Static files (CSS, JS)
│   ├── best_model.pth    # Trained model weights
│   ├── vocab.pkl         # Vocabulary mapping
│   └── label_mapping.pkl # Label mapping
├── train_multi.py        # Training script
├── test.csv             # Dataset
└── requirements.txt     # Project dependencies
```

## Setup and Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/SentimentSense.git
cd SentimentSense
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Training the Model

1. **Prepare the dataset**
   - The dataset should be in CSV format with 'text' and 'label' columns
   - Place your dataset as `test.csv` in the root directory

2. **Run the training script**
```bash
python train_multi.py
```

The training script will:
- Load and preprocess the data
- Build vocabulary from the training set
- Train the BiLSTM model
- Save the model and necessary files to the `flask` directory

## Model Architecture

### BiLSTM with Multi-head Attention

The model uses a sophisticated architecture combining:
1. **Embedding Layer**
   - Vocabulary size: 15,000
   - Embedding dimension: 300
   - Dropout: 0.4

2. **BiLSTM Layers**
   - 3 bidirectional LSTM layers
   - Hidden dimension: 256
   - Residual connections between layers
   - Dropout: 0.4

3. **Multi-head Attention**
   - 8 attention heads
   - Each head with LayerNorm and ReLU activation
   - Concatenated attention outputs

4. **Classification Head**
   - Deep neural network with residual connections
   - LayerNorm for regularization
   - Dropout for preventing overfitting

### Training Process

1. **Data Preprocessing**
   - Text tokenization
   - Vocabulary building
   - Sequence padding/truncation
   - Data augmentation:
     - Random word dropout
     - Random word swap

2. **Training Parameters**
   - Batch size: 8
   - Learning rate: 5e-5
   - Weight decay: 0.01
   - Maximum epochs: 200
   - Early stopping patience: 15
   - Gradient clipping: 1.0

3. **Class Imbalance Handling**
   - Class weights: 2.0x for minority classes
   - Stratified sampling
   - Data augmentation for minority classes

## Running the Flask Application

1. **Start the Flask server**
```bash
cd flask
python run.py
```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`
   - Enter text to classify
   - View the prediction results

## Model Performance

### Overall Metrics
- Overall accuracy: ~51%
- Macro average F1-score: 0.45
- Weighted average F1-score: 0.51

### Class-wise Performance
```
              precision    recall  f1-score   support
           0       0.62      0.62      0.62       116
           1       0.63      0.46      0.53       139
           2       0.35      0.22      0.27        32
           3       0.35      0.58      0.44        55
           4       0.46      0.49      0.47        45
           5       0.30      0.54      0.39        13
```

### Training Progress
- Best validation accuracy: 56.25% (achieved at epoch 66)
- Training accuracy reached: 77.88%
- Training duration: 68 epochs before early stopping

### Class Distribution
- Class 0: 116 samples (29%)
- Class 1: 139 samples (34.75%)
- Class 2: 32 samples (8%)
- Class 3: 55 samples (13.75%)
- Class 4: 45 samples (11.25%)
- Class 5: 13 samples (3.25%)

### Performance Analysis
1. **Strong Points**
   - High precision (0.62-0.63) for majority classes (0 and 1)
   - Good recall (0.54-0.58) for classes 3 and 5
   - Balanced performance across metrics

2. **Areas for Improvement**
   - Lower performance on minority classes (2 and 5)
   - Class 2 shows lowest F1-score (0.27)
   - Class 5 has low precision (0.30)

3. **Balancing Techniques Used**
   - Class weighting (2.0x for minority classes)
   - Data augmentation
   - Regularization techniques

## Technical Details

### Dependencies
- PyTorch
- Flask
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- tqdm

### File Descriptions

1. **train_multi.py**
   - Main training script
   - Implements model architecture
   - Handles data preprocessing
   - Manages training process

2. **flask/app.py**
   - Flask web application
   - Loads trained model
   - Handles text preprocessing
   - Returns predictions

3. **test.csv**
   - Training dataset
   - Contains text and labels
   - Used for model training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors
- Inspired by mental health text classification research
- Built with PyTorch and Flask
