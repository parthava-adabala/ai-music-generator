#!/usr/bin/env python3
"""
AI Music Generator using TensorFlow and ABC Notation
A neural network that learns to generate music in ABC notation format.
"""

import tensorflow as tf
import numpy as np
import os
import subprocess
import time
import argparse
from tqdm import tqdm
from scipy.io.wavfile import write
from midi2audio import FluidSynth
import matplotlib.pyplot as plt


class MusicGenerator:
    def __init__(self, data_path='data/input.txt'):
        self.data_path = data_path
        self.songs = []
        self.vocab = []
        self.char2idx = {}
        self.idx2char = None
        self.vectorized_songs = None
        self.model = None
        self.params = {
            'num_training_iterations': 1000,
            'batch_size': 16,
            'seq_length': 100,
            'learning_rate': 0.005,
            'embedding_dim': 256,
            'rnn_units': 1024
        }
        
    def _ensure_valid_abc(self, abc_text: str) -> str:
        """Ensure the ABC text has required headers and reasonable length/tempo.
        - Guarantees presence of X:, T:, M:, L:, Q:, K:
        - Ensures at least a few bars by repeating the body if extremely short
        - Normalizes empty or malformed bodies
        """
        # Split into lines and strip
        lines = [line.rstrip() for line in abc_text.splitlines() if line.strip()]

        # Extract existing headers and body
        header_keys = {"X:": None, "T:": None, "M:": None, "L:": None, "Q:": None, "K:": None}
        header_lines = []
        body_lines = []
        in_body = False
        for line in lines:
            key = line[:2]
            if not in_body and key in header_keys:
                header_keys[key] = line
                header_lines.append(line)
                if key == "K:":
                    in_body = True
            else:
                body_lines.append(line)

        # Defaults if missing
        defaults = {
            "X:": "X:1",
            "T:": "T:Generated",
            "M:": "M:4/4",
            "L:": "L:1/8",
            "Q:": "Q:1/4=120",
            "K:": "K:C",
        }
        normalized_headers = []
        for key in ["X:", "T:", "M:", "L:", "Q:", "K:"]:
            normalized_headers.append(header_keys[key] if header_keys[key] else defaults[key])

        # Basic body fallback if empty or too short
        if not body_lines:
            body_lines = ["|: CDEF GABc | d2 c2 B2 A2 | GABc d2 cB | A2 G2 F2 E2 :|"]

        # If body appears extremely short (few bars), repeat to reach at least ~8 bars
        bar_count = sum(line.count("|") for line in body_lines)
        if bar_count < 8:
            repeats_needed = max(1, (8 - bar_count) // max(1, bar_count)) if bar_count > 0 else 3
            body_lines = body_lines * (1 + repeats_needed)

        # Join back
        normalized = "\n".join(normalized_headers + body_lines)
        return normalized

    def load_training_data(self, file_path=None):
        """Load and preprocess the ABC music dataset."""
        if file_path is None:
            file_path = self.data_path
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Split into individual songs (songs are separated by 'X:')
            songs = [song.strip() for song in content.split('X:') if song.strip()]
            songs = ['X:' + song for song in songs]  # Re-add 'X:'
            self.songs = songs
            print(f'Loaded {len(songs)} songs.')
            return songs
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file {file_path} not found. Please verify the file path.")

    def vectorize_text(self):
        """Convert text to numerical representation."""
        songs_joined = '\n\n'.join(self.songs)
        self.vocab = sorted(set(songs_joined))
        print(f'There are {len(self.vocab)} unique characters in the dataset.')

        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        def vectorize_string(string):
            return np.array([self.char2idx[char] for char in string])

        self.vectorized_songs = vectorize_string(songs_joined)
        print(f'First 10 chars: {repr(songs_joined[:10])} -> {self.vectorized_songs[:10]}')

    def get_batch(self, seq_length, batch_size):
        """Create training batches from the vectorized songs."""
        n = self.vectorized_songs.shape[0] - 1
        idx = np.random.choice(n - seq_length, batch_size)
        input_batch = [self.vectorized_songs[i:i + seq_length] for i in idx]
        output_batch = [self.vectorized_songs[i + 1:i + 1 + seq_length] for i in idx]
        return np.array(input_batch), np.array(output_batch)

    def build_model(self, vocab_size, embedding_dim, rnn_units, batch_size):
        """Build the LSTM model for music generation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, 
                               recurrent_initializer='glorot_uniform',
                               recurrent_activation='sigmoid', stateful=True),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    def train_model(self, save_checkpoints=True):
        """Train the music generation model."""
        print("Building model...")
        self.model = self.build_model(len(self.vocab), 
                                    self.params['embedding_dim'], 
                                    self.params['rnn_units'], 
                                    self.params['batch_size'])
        self.model.summary()

        # Define loss and optimizer
        def compute_loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                y_hat = self.model(x)
                loss = compute_loss(y, y_hat)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        # Training loop
        history = []
        checkpoint_dir = './training_checkpoints'
        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'my_ckpt.weights.h5')

        print("Starting training...")
        for iter in tqdm(range(self.params['num_training_iterations'])):
            x_batch, y_batch = self.get_batch(self.params['seq_length'], self.params['batch_size'])
            loss = train_step(x_batch, y_batch)
            history.append(loss.numpy().mean())
            if iter % 100 == 0:
                if save_checkpoints:
                    self.model.save_weights(checkpoint_prefix)
                print(f'Iteration {iter}, Loss: {loss.numpy().mean():.4f}')

        if save_checkpoints:
            self.model.save_weights(checkpoint_prefix)

        # Plot and save training loss
        plt.figure(figsize=(10, 6))
        plt.plot(history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return history

    def load_trained_model(self, checkpoint_path='./training_checkpoints/my_ckpt.weights.h5'):
        """Load a pre-trained model for inference."""
        # Rebuild model for inference (batch_size=1)
        self.model = self.build_model(len(self.vocab), 
                                    self.params['embedding_dim'], 
                                    self.params['rnn_units'], 
                                    batch_size=1)
        self.model.build(tf.TensorShape([1, None]))
        self.model.load_weights(checkpoint_path)
        print("Model loaded successfully!")

    def generate_text(self, start_string, generation_length=1000, temperature=1.0):
        """Generate new music text starting from a given string."""
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        
        for _ in tqdm(range(generation_length)):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0) / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx2char[predicted_id])
            
        return start_string + ''.join(text_generated)

    def extract_song_snippet(self, text):
        """Extract individual songs from generated text."""
        songs = [song.strip() for song in text.split('X:') if song.strip()]
        valid_songs = []
        for song in songs:
            candidate = 'X:' + song
            # Must contain key and at least some note letters
            if 'K:' in candidate and any(ch in candidate for ch in 'ABCDEFGabcdefg'):
                # Normalize headers and ensure reasonable length
                valid_songs.append(self._ensure_valid_abc(candidate))
        return valid_songs

    def play_song(self, abc_text, output_file='temp.wav'):
        """Convert ABC notation to audio and save as WAV file."""
        try:
            # Write ABC to temp file
            abc_file = 'temp.abc'
            with open(abc_file, 'w', encoding='utf-8') as f:
                f.write(abc_text)

            # Convert ABC to MIDI
            midi_file = 'temp.mid'
            result = subprocess.run(['abc2midi', abc_file, '-o', midi_file], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error converting ABC to MIDI:\n{result.stderr}")
                return None

            # Convert MIDI to WAV using midi2audio with boosted gain
            # Note: gain range typically 0.0 - 10.0 (1.0 is default). Keep modest to avoid clipping.
            fs = FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2', sample_rate=44100, gain=1.2)
            fs.midi_to_audio(midi_file, output_file)

            if os.path.exists(output_file):
                print(f"Audio saved to {output_file}")
                return output_file
            return None
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting ABC to MIDI: {e}")
            return None
        except Exception as e:
            print(f"Error during MIDI to WAV conversion: {e}")
            return None

    def generate_and_save_music(self, start_strings=None, output_dir='generated_music', generation_length=2000, temperature=0.9):
        """Generate music and save as audio files."""
        if start_strings is None:
            start_strings = ['X:1\nT:Generated\nM:4/4\nL:1/8\nQ:1/4=120\nK:Em\n']
            
        os.makedirs(output_dir, exist_ok=True)
        
        for i, start in enumerate(start_strings):
            print(f"\nGenerating with start string: {repr(start)}")
            generated_text = self.generate_text(start_string=start, 
                                              generation_length=generation_length, 
                                              temperature=temperature)

            generated_songs = self.extract_song_snippet(generated_text)
            
            for j, song in enumerate(generated_songs):
                output_file = os.path.join(output_dir, f'generated_song_start{i}_song{j}.wav')
                result = self.play_song(song, output_file)
                if result:
                    print(f'Generated song {j} (start string {i}) saved to {output_file}')
                else:
                    print(f'Failed to generate audio for song {j} (start string {i})')


def main():
    parser = argparse.ArgumentParser(description='AI Music Generator')
    parser.add_argument('--data', default='data/input.txt', help='Path to training data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate music')
    parser.add_argument('--checkpoint', default='./training_checkpoints/my_ckpt.weights.h5', 
                       help='Path to model checkpoint')
    parser.add_argument('--output', default='generated_music', help='Output directory for generated music')
    parser.add_argument('--gen_length', type=int, default=2000, help='Generation length (characters)')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Initialize music generator
    generator = MusicGenerator(args.data)
    
    if args.train:
        # Load data and train model
        generator.load_training_data()
        generator.vectorize_text()
        generator.train_model()
        print("Training completed!")
        
    if args.generate:
        # Load trained model and generate music
        generator.load_training_data()
        generator.vectorize_text()
        generator.load_trained_model(args.checkpoint)
        generator.generate_and_save_music(output_dir=args.output, generation_length=args.gen_length, temperature=args.temperature)
        print("Music generation completed!")


if __name__ == "__main__":
    main()
