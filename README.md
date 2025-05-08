# SentimentSense

SentimentSense is an advanced emotion analysis tool that uses deep learning to analyze text and predict emotional content. Built with PyTorch and Flask, it provides real-time sentiment analysis with a beautiful web interface.

## Features

- **Multi-Emotion Detection**: Identifies 6 different emotional states:
  - Negative (sad, depressed, exhausted)
  - Positive (happy, excited, optimistic)
  - Affectionate/Caring
  - Angry/Frustrated
  - Anxious/Fearful
  - Surprised/Shocked

- **Advanced AI Model**: 
  - BiLSTM (Bidirectional Long Short-Term Memory) architecture
  - Multi-head attention mechanism
  - Deep neural network with multiple layers
  - Pre-trained on extensive emotional text data

- **User-Friendly Interface**:
  - Clean and modern web design
  - Real-time predictions
  - Inspirational quotes based on detected emotions
  - Responsive layout

## Technical Details

- **Model Architecture**:
  - Embedding Layer (300 dimensions)
  - 3 BiLSTM layers with attention
  - 8 attention heads
  - Multi-layer classifier
  - Dropout for regularization

- **Technologies Used**:
  - Python 3.10
  - PyTorch for deep learning
  - Flask for web framework
  - HTML/CSS for frontend
  - JavaScript for interactivity

## Getting Started

### Prerequisites

- Python 3.10
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SentimentSense.git
cd SentimentSense
```

2. Install the required packages:
```bash
pip install -r flask/requirements.txt
```

### Running the Application

1. Navigate to the flask directory:
```bash
cd flask
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and go to:
```
http://localhost:5000
```

## Usage

1. Enter your text in the input field
2. Click "Analyze" or press Enter
3. View the predicted emotion and related inspirational quote

## Project Structure

```
SentimentSense/
├── flask/
│   ├── app.py              # Main Flask application
│   ├── best_model.pth      # Trained model weights
│   ├── vocab.pkl          # Vocabulary file
│   ├── requirements.txt    # Python dependencies
│   ├── static/            # Static files (CSS, JS, images)
│   └── templates/         # HTML templates
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Inspired by the need for better emotional intelligence in text analysis
