# AI Music Generator - Makefile for easy commands

.PHONY: help build train generate clean docker-build docker-train docker-generate

# Default target
help:
	@echo "🎵 AI Music Generator - Available Commands:"
	@echo "=========================================="
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  docker-build     Build the Docker image"
	@echo "  docker-train     Train model using Docker"
	@echo "  docker-generate  Generate music using Docker"
	@echo "  docker-clean     Clean up Docker containers and images"
	@echo ""
	@echo "🐍 Local Commands:"
	@echo "  train           Train the model locally"
	@echo "  generate        Generate music locally"
	@echo "  clean           Clean generated files and checkpoints"
	@echo ""
	@echo "📦 Setup Commands:"
	@echo "  install         Install Python dependencies"
	@echo "  setup           Complete project setup"
	@echo ""

# Docker commands
docker-build:
	@echo "🐳 Building Docker image..."
	docker-compose build

docker-train:
	@echo "🎵 Training model with Docker..."
	docker-compose run --rm train

docker-generate:
	@echo "🎼 Generating music with Docker..."
	docker-compose run --rm generate

docker-clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down --rmi all --volumes --remove-orphans

# Local commands
train:
	@echo "🎵 Training model locally..."
	./scripts/train.sh

generate:
	@echo "🎼 Generating music locally..."
	./scripts/generate.sh

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf training_checkpoints/
	rm -rf generated_music/
	rm -f *.wav *.mid *.abc temp.*
	rm -f loss_plot.png

# Setup commands
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt

setup: install
	@echo "🔧 Setting up project directories..."
	mkdir -p data training_checkpoints generated_music
	@echo "✅ Project setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Add your training data to data/input.txt"
	@echo "2. Run 'make train' to train the model"
	@echo "3. Run 'make generate' to create music"

# Quick start for Docker users
quickstart: docker-build docker-train docker-generate
	@echo "🎉 Quick start complete! Check generated_music/ for your AI-generated songs!"
