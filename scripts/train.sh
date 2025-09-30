#!/bin/bash

# AI Music Generator - Training Script
# This script trains the neural network model

echo "🎵 Starting AI Music Generator Training..."
echo "=========================================="

# Check if data file exists
if [ ! -f "data/input.txt" ]; then
    echo "❌ Error: Training data not found at data/input.txt"
    echo "Please ensure the data file exists before training."
    exit 1
fi

# Create necessary directories
mkdir -p training_checkpoints
mkdir -p generated_music

echo "📊 Training the model..."
echo "This may take 30-60 minutes depending on your hardware."
echo ""

# Run training
python music_generator.py --train --data data/input.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📁 Model checkpoints saved to: training_checkpoints/"
    echo "📈 Training loss plot saved as: loss_plot.png"
    echo ""
    echo "🎶 Ready to generate music! Run:"
    echo "   python music_generator.py --generate"
    echo "   or"
    echo "   ./scripts/generate.sh"
else
    echo ""
    echo "❌ Training failed. Please check the error messages above."
    exit 1
fi
