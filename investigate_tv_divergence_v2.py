import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter

# Load the data
print("Loading data...")
df = pd.read_csv("TMDB_tv_dataset_v3.csv")
embeddings = np.load("tv_plot_embeddings.npy")
df['year'] = pd.to_datetime(df['first_air_date'], errors='coerce').dt.year
df['embedding_idx'] = range(len(df))

# Focus on the problematic period
years_to_check = [2015, 2018, 2020, 2021, 2022, 2023]

print("\n" + "=" * 80)
print("DEEP DIVE: What's in the embeddings?")
print("=" * 80)

for year in years_to_check:
    year_df = df[df['year'] == year]
    year_embeddings = embeddings[year_df['embedding_idx'].values]

    # Filter NaN
    valid_mask = ~np.isnan(year_embeddings).any(axis=1)
    year_embeddings_valid = year_embeddings[valid_mask]

    # Calculate norms
    norms = np.linalg.norm(year_embeddings_valid, axis=1)

    # Filter zeros
    non_zero_mask = norms > 0.01
    year_embeddings_nonzero = year_embeddings_valid[non_zero_mask]

    print(f"\n{year}:")
    print(f"  Total shows: {len(year_df)}")
    print(f"  After filtering zeros: {len(year_embeddings_nonzero)}")
    print(f"  Percentage kept: {100*len(year_embeddings_nonzero)/len(year_df):.1f}%")

    # Check for near-duplicates
    if len(year_embeddings_nonzero) > 1:
        # Look at pairwise distances to find clusters
        pairwise = cosine_distances(year_embeddings_nonzero)

        # Find all pairs with very low distance (likely duplicates or very similar)
        np.fill_diagonal(pairwise, 1.0)  # Ignore self-comparisons
        very_similar = (pairwise < 0.01).sum() / 2  # divide by 2 because symmetric

        print(f"  Near-duplicate pairs (dist < 0.01): {int(very_similar)}")

        # Check distribution of distances
        upper_tri = np.triu_indices_from(pairwise, k=1)
        distances = pairwise[upper_tri]

        print(f"  Distance distribution:")
        print(f"    Min: {distances.min():.4f}")
        print(f"    10th percentile: {np.percentile(distances, 10):.4f}")
        print(f"    25th percentile: {np.percentile(distances, 25):.4f}")
        print(f"    Median: {np.percentile(distances, 50):.4f}")
        print(f"    75th percentile: {np.percentile(distances, 75):.4f}")
        print(f"    90th percentile: {np.percentile(distances, 90):.4f}")
        print(f"    Max: {distances.max():.4f}")

        # Check how many are at exactly 1.0 (orthogonal)
        at_max = (distances > 0.999).sum()
        print(f"  Distances at/near 1.0 (orthogonal): {at_max} ({100*at_max/len(distances):.1f}%)")

print("\n" + "=" * 80)
print("CHECKING OVERVIEW QUALITY")
print("=" * 80)

for year in [2020, 2021, 2022, 2023]:
    year_df = df[df['year'] == year].copy()

    # Add overview length
    year_df['overview_length'] = year_df['overview'].fillna('').str.len()

    # Get embeddings and norms
    year_embeddings = embeddings[year_df['embedding_idx'].values]
    norms = np.linalg.norm(year_embeddings, axis=1)
    year_df['embedding_norm'] = norms

    # Filter to non-zero embeddings
    non_zero = year_df[norms > 0.01]

    print(f"\n{year}:")
    print(f"  Overview length stats (non-zero embeddings):")
    print(f"    Min: {non_zero['overview_length'].min()}")
    print(f"    25th: {non_zero['overview_length'].quantile(0.25):.0f}")
    print(f"    Median: {non_zero['overview_length'].median():.0f}")
    print(f"    75th: {non_zero['overview_length'].quantile(0.75):.0f}")
    print(f"    Max: {non_zero['overview_length'].max()}")
    print(f"  Very short overviews (<50 chars): {(non_zero['overview_length'] < 50).sum()}")

    # Sample some short overviews
    short_overviews = non_zero[non_zero['overview_length'] < 100].sample(min(5, len(non_zero[non_zero['overview_length'] < 100])))
    if len(short_overviews) > 0:
        print(f"  Sample short overviews:")
        for idx, row in short_overviews.iterrows():
            print(f"    - ({row['overview_length']} chars): {row['overview'][:100]}")

print("\n" + "=" * 80)
print("HYPOTHESIS: Are recent shows embedding to similar vectors?")
print("=" * 80)

# Check if there's clustering in recent years
for year in [2018, 2020, 2023]:
    year_df = df[df['year'] == year]
    year_embeddings = embeddings[year_df['embedding_idx'].values]

    # Filter properly
    valid_mask = ~np.isnan(year_embeddings).any(axis=1)
    year_embeddings = year_embeddings[valid_mask]
    norms = np.linalg.norm(year_embeddings, axis=1)
    year_embeddings = year_embeddings[norms > 0.01]

    if len(year_embeddings) < 100:
        continue

    # Sample 1000 for speed
    if len(year_embeddings) > 1000:
        indices = np.random.choice(len(year_embeddings), 1000, replace=False)
        sample_embeddings = year_embeddings[indices]
    else:
        sample_embeddings = year_embeddings

    # Calculate mean pairwise distance
    pairwise = cosine_distances(sample_embeddings)
    upper_tri = np.triu_indices_from(pairwise, k=1)
    mean_dist = pairwise[upper_tri].mean()

    # Calculate distance from centroid
    centroid = sample_embeddings.mean(axis=0)
    dist_from_centroid = cosine_distances(sample_embeddings, centroid.reshape(1, -1)).mean()

    # Calculate ratio
    ratio = mean_dist / dist_from_centroid if dist_from_centroid > 0 else 0

    print(f"\n{year} (sample n={len(sample_embeddings)}):")
    print(f"  Mean pairwise distance: {mean_dist:.4f}")
    print(f"  Mean distance from centroid: {dist_from_centroid:.4f}")
    print(f"  Ratio (pairwise/centroid): {ratio:.4f}")
    print(f"  Expected ratio for uniform distribution: ~1.29")
    print(f"  Interpretation: Ratio > 1.29 suggests clustering/outliers")

print("\n" + "=" * 80)
print("Done!")
