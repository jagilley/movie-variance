import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances

# Load the data
print("Loading data...")
df = pd.read_csv("wiki_movie_plots_deduped.csv")
embeddings = np.load("movie_plot_embeddings.npy")

print(f"Loaded {len(df)} movies and {len(embeddings)} embeddings")

# Add embeddings to dataframe for easier grouping
df['embedding_idx'] = range(len(df))

# Group by year and calculate variance metrics
print("\nCalculating variance metrics by year...")
results = []

for year in sorted(df['Release Year'].unique()):
    year_df = df[df['Release Year'] == year]

    # Skip years with too few movies (need at least 2 for variance)
    if len(year_df) < 2:
        continue

    # Get embeddings for this year
    year_embeddings = embeddings[year_df['embedding_idx'].values]

    # Filter out any NaN embeddings
    valid_mask = ~np.isnan(year_embeddings).any(axis=1)
    year_embeddings = year_embeddings[valid_mask]

    # Skip if we don't have enough valid embeddings
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
        'year': year,
        'num_movies': len(year_embeddings),
        'avg_pairwise_cosine_dist': avg_pairwise_cosine_dist,
        'avg_dist_from_centroid': avg_dist_from_centroid,
        'std_across_dims': std_across_dims
    })

results_df = pd.DataFrame(results)

# Print summary statistics
print("\nSummary Statistics:")
print(f"Years analyzed: {len(results_df)}")
print(f"Total movies: {results_df['num_movies'].sum()}")
print(f"\nCorrelation between year and variance metrics:")
print(f"  Avg Pairwise Cosine Distance: {results_df['year'].corr(results_df['avg_pairwise_cosine_dist']):.4f}")
print(f"  Avg Distance from Centroid: {results_df['year'].corr(results_df['avg_dist_from_centroid']):.4f}")
print(f"  Std Across Dimensions: {results_df['year'].corr(results_df['std_across_dims']):.4f}")

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Average Pairwise Cosine Distance
axes[0].scatter(results_df['year'], results_df['avg_pairwise_cosine_dist'],
                alpha=0.6, s=results_df['num_movies']*2, c=results_df['num_movies'],
                cmap='viridis')
axes[0].plot(results_df['year'], results_df['avg_pairwise_cosine_dist'],
             alpha=0.3, linewidth=1, color='red')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Average Pairwise Cosine Distance')
axes[0].set_title('Variance in Movie Plots Over Time (Pairwise Cosine Distance)')
axes[0].grid(True, alpha=0.3)

# Plot 2: Average Distance from Centroid
axes[1].scatter(results_df['year'], results_df['avg_dist_from_centroid'],
                alpha=0.6, s=results_df['num_movies']*2, c=results_df['num_movies'],
                cmap='viridis')
axes[1].plot(results_df['year'], results_df['avg_dist_from_centroid'],
             alpha=0.3, linewidth=1, color='red')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Distance from Centroid')
axes[1].set_title('Variance in Movie Plots Over Time (Distance from Centroid)')
axes[1].grid(True, alpha=0.3)

# Plot 3: Standard Deviation Across Dimensions
axes[2].scatter(results_df['year'], results_df['std_across_dims'],
                alpha=0.6, s=results_df['num_movies']*2, c=results_df['num_movies'],
                cmap='viridis')
axes[2].plot(results_df['year'], results_df['std_across_dims'],
             alpha=0.3, linewidth=1, color='red')
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Mean Std Dev Across Dimensions')
axes[2].set_title('Variance in Movie Plots Over Time (Standard Deviation)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Add colorbar below all plots
cbar = fig.colorbar(axes[0].collections[0], ax=axes, orientation='horizontal',
                    pad=0.08, aspect=40, shrink=0.8)
cbar.set_label('Number of Movies in Year')
plt.savefig('variance_over_time.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: variance_over_time.png")

# Save detailed results
results_df.to_csv('variance_by_year.csv', index=False)
print(f"✓ Detailed results saved to: variance_by_year.csv")
