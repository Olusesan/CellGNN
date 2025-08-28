import numpy as np
from scipy.spatial.distance import cdist

def create_adaptive_temporal_graph(features_t1, features_t2, max_distance=50, 
                                 adaptive_threshold=True, feature_weights=None):
    """Create graph with adaptive connection strategy and weighted features"""
    if len(features_t1) == 0 or len(features_t2) == 0:
        return np.empty((2, 0), dtype=int), np.empty(0)

    centroids_t1 = features_t1[:, :2]
    centroids_t2 = features_t2[:, :2]

    # Calculate distance matrix (unnormalized positions)
    distances = cdist(centroids_t1 * np.array([features_t1.shape[1], features_t1.shape[0]]), 
                     centroids_t2 * np.array([features_t2.shape[1], features_t2.shape[0]]))

    # Adaptive distance threshold based on cell density
    if adaptive_threshold:
        cell_density = (len(features_t1) + len(features_t2)) / 2
        density_factor = min(2.0, max(0.5, 100 / cell_density))
        effective_max_distance = max_distance * density_factor
    else:
        effective_max_distance = max_distance

    # Feature similarity with optional weighting
    if feature_weights is None:
        feature_weights = np.ones(features_t1.shape[1])
    
    # Weight features for similarity calculation
    weighted_features_t1 = features_t1 * feature_weights
    weighted_features_t2 = features_t2 * feature_weights
    
    # Robust feature similarity
    feature_similarities = []
    for i in range(len(features_t1)):
        for j in range(len(features_t2)):
            # Euclidean distance in feature space
            feat_dist = np.linalg.norm(weighted_features_t1[i] - weighted_features_t2[j])
            max_feat_dist = np.linalg.norm(weighted_features_t1.std(axis=0) + weighted_features_t2.std(axis=0))
            similarity = 1 - (feat_dist / (max_feat_dist + 1e-8))
            feature_similarities.append(max(0, similarity))
    
    feature_similarities = np.array(feature_similarities).reshape(len(features_t1), len(features_t2))

    # Combined score with balanced weighting
    max_dist = distances.max() + 1e-8
    normalized_distances = distances / max_dist
    combined_scores = 0.6 * (1 - normalized_distances) + 0.4 * feature_similarities

    edges = []
    labels = []

    # Create edges with improved strategy
    for i in range(len(features_t1)):
        # Get candidates within distance threshold
        valid_mask = distances[i] <= effective_max_distance
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            # Sort by combined score
            scores = combined_scores[i][valid_indices]
            sorted_idx = np.argsort(scores)[::-1]
            sorted_indices = valid_indices[sorted_idx]

            # Dynamic connection limit based on score distribution
            score_threshold = scores.max() * 0.7  # Only keep high-scoring connections
            good_connections = scores >= score_threshold
            
            max_connections = min(3, max(1, good_connections.sum()))

            # Create edges for top connections
            for k, j in enumerate(sorted_indices[:max_connections]):
                edges.append([i, len(features_t1) + j])
                
                # More nuanced labeling
                score_ratio = scores[sorted_idx[k]] / (scores.max() + 1e-8)
                label = 1.0 if (k == 0 and score_ratio > 0.8) else 0.0
                labels.append(label)
    
    unique_labels = np.array(labels)
    sorted_edges = np.array(edges).T

    if len(edges) == 0:
        return np.empty((2, 0), dtype=int), np.empty(0)

    return sorted_edges, unique_labels