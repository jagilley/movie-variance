import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances

# Load the data
print("Loading data...")
df = pd.read_csv("TMDB_tv_dataset_v3.csv")
embeddings = np.load("tv_plot_embeddings.npy")

# Extract year from first_air_date
df['year'] = pd.to_datetime(df['first_air_date'], errors='coerce').dt.year

print(f"Loaded {len(df)} TV shows and {len(embeddings)} embeddings")

# FILTER 1: Remove shows with short overviews (< 50 characters)
df['overview_length'] = df['overview'].fillna('').str.len()
df_filtered = df[df['overview_length'] >= 50].copy()
print(f"After filtering short overviews (<50 chars): {len(df_filtered)} shows")

# Add embeddings to dataframe for easier grouping
df_filtered['embedding_idx'] = df_filtered.index

# Group by year and calculate variance metrics
print("\nCalculating variance metrics by year...")
print("(Applying deduplication and quality filters...)")
results = []
total_duplicates_removed = 0

for year in sorted(df_filtered['year'].dropna().unique()):
    year_df = df_filtered[df_filtered['year'] == year]

    # Skip years with too few TV shows (need at least 2 for variance)
    if len(year_df) < 2:
        continue

    # Get embeddings for this year
    year_embeddings = embeddings[year_df['embedding_idx'].values]

    # FILTER 2: Filter out any NaN embeddings
    valid_mask = ~np.isnan(year_embeddings).any(axis=1)
    year_embeddings = year_embeddings[valid_mask]

    # FILTER 3: Filter out zero/near-zero embeddings
    embedding_norms = np.linalg.norm(year_embeddings, axis=1)
    non_zero_mask = embedding_norms > 0.01
    year_embeddings = year_embeddings[non_zero_mask]

    # Skip if we don't have enough valid embeddings
    if len(year_embeddings) < 2:
        continue

    # FILTER 4: Remove near-duplicates (cosine distance < 0.01)
    # This is computationally expensive, so we do it efficiently
    pairwise_dist = cosine_distances(year_embeddings)
    np.fill_diagonal(pairwise_dist, 1.0)  # Ignore self-comparisons

    # Find duplicates: for each embedding, check if any other is too close
    to_remove = set()
    for i in range(len(year_embeddings)):
        if i in to_remove:
            continue
        # Find all embeddings very similar to this one
        similar = np.where(pairwise_dist[i] < 0.01)[0]
        # Remove all but the first occurrence
        for j in similar:
            if j > i:  # Only remove later indices to avoid conflicts
                to_remove.add(j)

    # Keep only non-duplicate embeddings
    if len(to_remove) > 0:
        keep_mask = np.ones(len(year_embeddings), dtype=bool)
        keep_mask[list(to_remove)] = False
        year_embeddings = year_embeddings[keep_mask]
        total_duplicates_removed += len(to_remove)

    # Skip if we don't have enough after deduplication
    if len(year_embeddings) < 2:
        continue

    # Calculate multiple variance metrics:

    # 1. Average pairwise cosine distance
    pairwise_cosine_dist = cosine_distances(year_embeddings)
    # Take upper triangle (excluding diagonal) to avoid double counting
    upper_tri_indices = np.triu_indices_from(pairwise_cosine_dist, k=1)
    avg_pairwise_cosine_dist = pairwise_cosine_dist[upper_tri_indices].mean()

    # 2. Average distance from centroid (using cosine distance)
    centroid = year_embeddings.mean(axis=0)
    distances_from_centroid = cosine_distances(year_embeddings, centroid.reshape(1, -1))
    avg_dist_from_centroid = distances_from_centroid.mean()

    # 3. Standard deviation across all dimensions
    std_across_dims = year_embeddings.std(axis=0).mean()

    results.append({
        'year': int(year),
        'num_shows': len(year_embeddings),
        'avg_pairwise_cosine_dist': avg_pairwise_cosine_dist,
        'avg_dist_from_centroid': avg_dist_from_centroid,
        'std_across_dims': std_across_dims
    })

results_df = pd.DataFrame(results)

# Print summary statistics
print(f"\nTotal near-duplicates removed: {total_duplicates_removed}")
print("\nSummary Statistics:")
print(f"Years analyzed: {len(results_df)}")
print(f"Total TV shows: {results_df['num_shows'].sum()}")
print(f"\nCorrelation between year and variance metrics:")
print(f"  Avg Pairwise Cosine Distance: {results_df['year'].corr(results_df['avg_pairwise_cosine_dist']):.4f}")
print(f"  Avg Distance from Centroid: {results_df['year'].corr(results_df['avg_dist_from_centroid']):.4f}")
print(f"  Std Across Dimensions: {results_df['year'].corr(results_df['std_across_dims']):.4f}")

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Average Pairwise Cosine Distance
axes[0].scatter(results_df['year'], results_df['avg_pairwise_cosine_dist'],
                alpha=0.6, s=results_df['num_shows']*2, c=results_df['num_shows'],
                cmap='viridis')
axes[0].plot(results_df['year'], results_df['avg_pairwise_cosine_dist'],
             alpha=0.3, linewidth=1, color='red')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Average Pairwise Cosine Distance')
axes[0].set_title('Variance in TV Show Plots Over Time (Pairwise Cosine Distance)')
axes[0].grid(True, alpha=0.3)

# Plot 2: Average Distance from Centroid
axes[1].scatter(results_df['year'], results_df['avg_dist_from_centroid'],
                alpha=0.6, s=results_df['num_shows']*2, c=results_df['num_shows'],
                cmap='viridis')
axes[1].plot(results_df['year'], results_df['avg_dist_from_centroid'],
             alpha=0.3, linewidth=1, color='red')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Distance from Centroid')
axes[1].set_title('Variance in TV Show Plots Over Time (Distance from Centroid)')
axes[1].grid(True, alpha=0.3)

# Plot 3: Standard Deviation Across Dimensions
axes[2].scatter(results_df['year'], results_df['std_across_dims'],
                alpha=0.6, s=results_df['num_shows']*2, c=results_df['num_shows'],
                cmap='viridis')
axes[2].plot(results_df['year'], results_df['std_across_dims'],
             alpha=0.3, linewidth=1, color='red')
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Mean Std Dev Across Dimensions')
axes[2].set_title('Variance in TV Show Plots Over Time (Standard Deviation)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Add colorbar below all plots
cbar = fig.colorbar(axes[0].collections[0], ax=axes, orientation='horizontal',
                    pad=0.08, aspect=40, shrink=0.8)
cbar.set_label('Number of TV Shows in Year')
plt.savefig('variance_over_time_tv.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: variance_over_time_tv.png")

# Save detailed results
results_df.to_csv('variance_by_year_tv.csv', index=False)
print(f"✓ Detailed results saved to: variance_by_year_tv.csv")
