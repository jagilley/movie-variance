import pandas as pd
import numpy as np
from mlx_embeddings.utils import load
from tqdm import tqdm
import mlx.core as mx

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv("wiki_movie_plots_deduped.csv")
print(f"Loaded {len(df)} movie plots")

# Load the model and tokenizer
print("Loading model and tokenizer...")
model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
model, tokenizer = load(model_name)
print("Model loaded successfully!")

# Function to generate embedding for a single text
def generate_embedding(text):
    """Generate embedding for a single text string"""
    if pd.isna(text) or text == "":
        # Return zero vector for empty plots
        return np.zeros(384)  # all-MiniLM-L6-v2 produces 384-dim embeddings

    # Tokenize and generate embedding
    input_ids = tokenizer.encode(text, return_tensors="mlx")
    outputs = model(input_ids)

    # Use mean pooled and normalized embeddings
    text_embeds = outputs.text_embeds

    # Convert to numpy
    return np.array(text_embeds[0])

# Generate embeddings for all plots
print("\nGenerating embeddings...")
embeddings = []

for plot in tqdm(df['Plot'], desc="Processing plots", unit="plot"):
    embedding = generate_embedding(plot)
    embeddings.append(embedding)

# Convert to numpy array
embeddings_array = np.array(embeddings)

# Save the embeddings
output_file = "movie_plot_embeddings.npy"
print(f"\nSaving embeddings to {output_file}...")
np.save(output_file, embeddings_array)

print(f"✓ Successfully generated {len(embeddings_array)} embeddings")
print(f"✓ Embedding shape: {embeddings_array.shape}")
print(f"✓ Saved to: {output_file}")
