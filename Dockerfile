# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    abcmidi \
    timidity \
    fluidsynth \
    fluid-soundfont-gm \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for output
RUN mkdir -p generated_music training_checkpoints

# Make the script executable
RUN chmod +x music_generator.py

# Set environment variables
ENV PYTHONPATH=/app
ENV FLUID_SYNTH_SOUNDFONT=/usr/share/sounds/sf2/FluidR3_GM.sf2

# Default command
CMD ["python", "music_generator.py", "--help"]
