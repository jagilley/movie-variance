import pandas as pd
import numpy as np
from mlx_embeddings.utils import load
from tqdm import tqdm
import mlx.core as mx

# Load the CSV file
print("Loading TV dataset...")
df = pd.read_csv("TMDB_tv_dataset_v3.csv")
print(f"Loaded {len(df)} TV shows")

# Extract year from first_air_date
df['year'] = pd.to_datetime(df['first_air_date'], errors='coerce').dt.year
print(f"TV shows with valid years: {df['year'].notna().sum()}")

# Load the model and tokenizer
print("Loading model and tokenizer...")
model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
model, tokenizer = load(model_name)
print("Model loaded successfully!")

# Function to generate embedding for a single text
def generate_embedding(text):
    """Generate embedding for a single text string"""
    if pd.isna(text) or text == "":
        # Return zero vector for empty overviews
        return np.zeros(384)  # all-MiniLM-L6-v2 produces 384-dim embeddings

    # Tokenize and generate embedding
    input_ids = tokenizer.encode(text, return_tensors="mlx")
    outputs = model(input_ids)

    # Use mean pooled and normalized embeddings
    text_embeds = outputs.text_embeds

    # Convert to numpy
    return np.array(text_embeds[0])

# Generate embeddings for all TV show overviews
print("\nGenerating embeddings...")
embeddings = []

for overview in tqdm(df['overview'], desc="Processing TV shows", unit="show"):
    embedding = generate_embedding(overview)
    embeddings.append(embedding)

# Convert to numpy array
embeddings_array = np.array(embeddings)

# Save the embeddings
output_file = "tv_plot_embeddings.npy"
print(f"\nSaving embeddings to {output_file}...")
np.save(output_file, embeddings_array)

print(f"✓ Successfully generated {len(embeddings_array)} embeddings")
print(f"✓ Embedding shape: {embeddings_array.shape}")
print(f"✓ Saved to: {output_file}")
