# 🎵 AI Music Generator

An intelligent music generation system that uses TensorFlow and LSTM neural networks to create original music in ABC notation format. The AI learns from traditional folk music patterns and generates new compositions that can be converted to audio files.

## 🌟 Features

- **Neural Network Training**: LSTM-based model that learns from ABC notation music
- **Music Generation**: Creates original compositions in ABC notation
- **Audio Conversion**: Converts generated music to WAV audio files
- **Docker Support**: Easy deployment with Docker containers
- **Portfolio Ready**: Professional documentation and examples

## 🚀 Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available for training

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd ai-music-generator
```

### 2. Train the Model
```bash
# Train the AI model (this may take 30-60 minutes)
docker-compose run --rm train
```

### 3. Generate Music
```bash
# Generate new music compositions
docker-compose run --rm generate
```

### 4. Access Generated Music
The generated WAV files will be available in the `generated_music/` directory.

## 🛠️ Manual Installation

### Prerequisites
- Python 3.9+
- TensorFlow 2.10+
- Audio processing tools (abcmidi, fluidsynth)

### Install Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y abcmidi timidity fluidsynth fluid-soundfont-gm
pip install -r requirements.txt
```

#### macOS:
```bash
brew install abcmidi timidity fluidsynth
pip install -r requirements.txt
```

#### Windows:
```bash
# Install via conda or use WSL
conda install -c conda-forge abcmidi fluidsynth
pip install -r requirements.txt
```

### Usage

#### Train the Model:
```bash
python music_generator.py --train --data data/input.txt
```

#### Generate Music:
```bash
python music_generator.py --generate --data data/input.txt --output generated_music
```

## 📁 Project Structure

```
ai-music-generator/
├── music_generator.py      # Main application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── data/
│   └── input.txt          # Training data (ABC notation)
├── generated_music/       # Output directory for generated audio
├── training_checkpoints/  # Saved model weights
└── README.md             # This file
```

## 🎼 ABC Notation

This project uses ABC notation, a text-based music notation system. Example:

```
X:1
T: The Enchanted Valley
M: 2/4
L: 1/16
K:Gm
G3-A (Bcd=e) | f4 (g2dB) | ({d}c3-B) G2-E2 | F4 (D2=E^F) |
```

## 🧠 How It Works

1. **Data Loading**: Loads ABC notation music from text files
2. **Text Vectorization**: Converts characters to numerical representations
3. **Model Training**: LSTM neural network learns patterns from the music
4. **Music Generation**: AI generates new sequences based on learned patterns
5. **Audio Conversion**: ABC notation is converted to MIDI, then to WAV audio

## ⚙️ Configuration

You can modify training parameters in `music_generator.py`:

```python
params = {
    'num_training_iterations': 1000,  # Training iterations
    'batch_size': 16,                 # Batch size
    'seq_length': 100,                # Sequence length
    'learning_rate': 0.005,           # Learning rate
    'embedding_dim': 256,             # Embedding dimension
    'rnn_units': 1024                 # LSTM units
}
```

## 🎯 Command Line Options

```bash
python music_generator.py [OPTIONS]

Options:
  --data PATH        Path to training data (default: data/input.txt)
  --train           Train the model
  --generate        Generate music
  --checkpoint PATH Path to model checkpoint
  --output PATH     Output directory for generated music
```

## 📊 Training Progress

The training process includes:
- Real-time loss monitoring
- Automatic checkpoint saving
- Training loss visualization
- Progress bars for long operations

## 🎵 Generated Music Examples

After training, the AI can generate music with different starting patterns:

- **Simple start**: `X:` - Generates from basic notation
- **Structured start**: `X:1\nT:Generated\nM:4/4\nK:Em\n` - Generates with specific structure

## 🔧 Troubleshooting

### Common Issues:

1. **Audio conversion fails**: Ensure fluidsynth and soundfonts are installed
2. **Memory errors**: Reduce batch_size or seq_length in parameters
3. **Docker build fails**: Check Docker has sufficient resources allocated

### Performance Tips:

- Use GPU acceleration if available (modify Dockerfile)
- Increase training iterations for better quality
- Adjust temperature parameter for generation creativity

## 📈 Portfolio Integration

This project demonstrates:
- **Machine Learning**: LSTM neural networks for sequence generation
- **Audio Processing**: ABC to MIDI to WAV conversion pipeline
- **Docker**: Containerized deployment for reproducibility
- **Software Engineering**: Clean code structure and documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- ABC notation community for the music format
- TensorFlow team for the machine learning framework
- Traditional folk music for the training data inspiration

---

**Ready to create AI-generated music? Start with Docker and let the AI compose for you! 🎶**
