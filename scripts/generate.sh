#!/bin/bash

# AI Music Generator - Generation Script
# This script generates new music using the trained model

echo "🎵 Generating AI Music..."
echo "========================="

# Check if model exists
if [ ! -f "training_checkpoints/my_ckpt.weights.h5" ]; then
    echo "❌ Error: Trained model not found!"
    echo "Please train the model first by running:"
    echo "   python music_generator.py --train"
    echo "   or"
    echo "   ./scripts/train.sh"
    exit 1
fi

# Check if data file exists
if [ ! -f "data/input.txt" ]; then
    echo "❌ Error: Training data not found at data/input.txt"
    exit 1
fi

# Create output directory
mkdir -p generated_music

echo "🎼 Generating new music compositions..."
echo ""

# Run generation
python music_generator.py --generate --data data/input.txt --output generated_music

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Music generation completed!"
    echo "🎵 Generated audio files saved to: generated_music/"
    echo ""
    echo "📁 Generated files:"
    ls -la generated_music/*.wav 2>/dev/null || echo "No WAV files found"
    echo ""
    echo "🎧 You can now play the generated music files!"
else
    echo ""
    echo "❌ Generation failed. Please check the error messages above."
    exit 1
fi
