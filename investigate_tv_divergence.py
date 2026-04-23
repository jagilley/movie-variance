import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

# Load the data
print("Loading data...")
df = pd.read_csv("TMDB_tv_dataset_v3.csv")
embeddings = np.load("tv_plot_embeddings.npy")
df['year'] = pd.to_datetime(df['first_air_date'], errors='coerce').dt.year
df['embedding_idx'] = range(len(df))

# Focus on recent years to understand the divergence
recent_years = [2015, 2018, 2020, 2022, 2023]

print("\nAnalyzing distribution shape for key years:")
print("=" * 80)

for year in recent_years:
    year_df = df[df['year'] == year]

    if len(year_df) < 2:
        continue

    # Get embeddings
    year_embeddings = embeddings[year_df['embedding_idx'].values]

    # Filter NaN
    valid_mask = ~np.isnan(year_embeddings).any(axis=1)
    year_embeddings = year_embeddings[valid_mask]

    if len(year_embeddings) < 2:
        continue

    # Calculate metrics
    pairwise_dist = cosine_distances(year_embeddings)
    upper_tri = np.triu_indices_from(pairwise_dist, k=1)

    centroid = year_embeddings.mean(axis=0)
    dist_from_centroid = cosine_distances(year_embeddings, centroid.reshape(1, -1)).flatten()

    std_across_dims = year_embeddings.std(axis=0).mean()

    # Key insight: look at the distribution of distances
    pairwise_distances_flat = pairwise_dist[upper_tri]

    print(f"\nYear {year} (n={len(year_embeddings)}):")
    print(f"  Pairwise distance: mean={pairwise_distances_flat.mean():.4f}, std={pairwise_distances_flat.std():.4f}")
    print(f"  Pairwise distance percentiles: [25%={np.percentile(pairwise_distances_flat, 25):.4f}, "
          f"50%={np.percentile(pairwise_distances_flat, 50):.4f}, "
          f"75%={np.percentile(pairwise_distances_flat, 75):.4f}, "
          f"95%={np.percentile(pairwise_distances_flat, 95):.4f}]")
    print(f"  Distance from centroid: mean={dist_from_centroid.mean():.4f}, std={dist_from_centroid.std():.4f}")
    print(f"  Distance from centroid percentiles: [25%={np.percentile(dist_from_centroid, 25):.4f}, "
          f"50%={np.percentile(dist_from_centroid, 50):.4f}, "
          f"75%={np.percentile(dist_from_centroid, 75):.4f}, "
          f"95%={np.percentile(dist_from_centroid, 95):.4f}]")
    print(f"  Std across dims: {std_across_dims:.4f}")

    # Check for outliers or clustering
    # If there are outliers, we'd see high percentile values for pairwise distances
    # If there's clustering, we'd see bimodal distributions

# Now let's visualize the distribution of distances for 2020-2023
print("\n" + "=" * 80)
print("Creating distribution visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Distances: Understanding the Metric Divergence', fontsize=14)

for idx, year in enumerate([2020, 2021, 2022, 2023]):
    if idx >= 4:
        break

    year_df = df[df['year'] == year]
    year_embeddings = embeddings[year_df['embedding_idx'].values]
    valid_mask = ~np.isnan(year_embeddings).any(axis=1)
    year_embeddings = year_embeddings[valid_mask]

    if len(year_embeddings) < 2:
        continue

    # Calculate both metrics
    pairwise_dist = cosine_distances(year_embeddings)
    upper_tri = np.triu_indices_from(pairwise_dist, k=1)
    pairwise_distances_flat = pairwise_dist[upper_tri]

    centroid = year_embeddings.mean(axis=0)
    dist_from_centroid = cosine_distances(year_embeddings, centroid.reshape(1, -1)).flatten()

    # Sample if too many points (for visualization)
    max_samples = 100000
    if len(pairwise_distances_flat) > max_samples:
        pairwise_sample = np.random.choice(pairwise_distances_flat, max_samples, replace=False)
    else:
        pairwise_sample = pairwise_distances_flat

    # Plot histograms
    ax = axes[idx // 2, idx % 2]
    ax.hist(pairwise_sample, bins=50, alpha=0.5, label='Pairwise distances', color='blue', density=True)
    ax.hist(dist_from_centroid, bins=50, alpha=0.5, label='Distance from centroid', color='red', density=True)
    ax.set_xlabel('Cosine Distance')
    ax.set_ylabel('Density')
    ax.set_title(f'{year} (n={len(year_embeddings)} shows)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add vertical lines for means
    ax.axvline(pairwise_sample.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(dist_from_centroid.mean(), color='red', linestyle='--', linewidth=2, alpha=0.8)

plt.tight_layout()
plt.savefig('tv_distance_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tv_distance_distributions.png")

# Additional insight: check if embedding quality degraded
print("\n" + "=" * 80)
print("Checking for potential data quality issues:")

for year in [2020, 2021, 2022, 2023]:
    year_df = df[df['year'] == year]
    year_embeddings = embeddings[year_df['embedding_idx'].values]

    # Check for zero/near-zero embeddings (could indicate empty/missing overviews)
    embedding_norms = np.linalg.norm(year_embeddings, axis=1)
    near_zero = np.sum(embedding_norms < 0.01)
    all_zero = np.sum(embedding_norms == 0)

    # Check for duplicate embeddings
    unique_embeddings = np.unique(year_embeddings, axis=0)

    print(f"\nYear {year} (n={len(year_df)}):")
    print(f"  Near-zero embeddings: {near_zero} ({100*near_zero/len(year_df):.2f}%)")
    print(f"  All-zero embeddings: {all_zero} ({100*all_zero/len(year_df):.2f}%)")
    print(f"  Unique embeddings: {len(unique_embeddings)} ({100*len(unique_embeddings)/len(year_df):.2f}%)")

    # Check for missing overviews
    missing_overview = year_df['overview'].isna().sum()
    empty_overview = (year_df['overview'] == '').sum()
    print(f"  Missing overviews: {missing_overview} ({100*missing_overview/len(year_df):.2f}%)")
    print(f"  Empty overviews: {empty_overview} ({100*empty_overview/len(year_df):.2f}%)")

print("\n" + "=" * 80)
print("Analysis complete!")
