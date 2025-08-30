import os
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread
from cellpose import models
from cell_features import extract_comprehensive_features
from graph_builder import create_adaptive_temporal_graph
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling



def create_comprehensive_cell_record(frame_idx, cell_idx, feature_row):
    """Create comprehensive cell record with all features"""
    return {
        'frame': frame_idx,
        'cell_id': f"f{frame_idx}_c{cell_idx}",
        'x': feature_row[0],
        'y': feature_row[1],
        'area': feature_row[2],
        'mean_intensity': feature_row[3],
        'max_intensity': feature_row[4],
        'perimeter': feature_row[5],
        'eccentricity': feature_row[6],
        'solidity': feature_row[7],
        'aspect_ratio': feature_row[8],
        'equivalent_diameter': feature_row[9] if len(feature_row) > 9 else 0,
        'major_axis_length': feature_row[10] if len(feature_row) > 10 else 0,
        'compactness': feature_row[11] if len(feature_row) > 11 else 0
    }


def load_and_process_comprehensive_data(folder_path=None, max_frames=50, negative_ratio=0.5, min_positive_ratio=0.25):
    """Enhanced data loading with comprehensive feature extraction"""
    if folder_path is None:
        folder_path = './01'

    # Load images
    sorted_files = sorted(glob(os.path.join(folder_path, '*.tif')))[:max_frames]

    if len(sorted_files) == 0:
        print(f"No .tif files found in {folder_path}")
        return [], []

    print(f"Loading {len(sorted_files)} images...")
    imgs = []
    for f in sorted_files:
        img = imread(f)
        if img.ndim > 2:
            img = np.mean(img, axis=-1) if img.shape[-1] == 3 else img.squeeze()
        imgs.append(img.astype(np.float32))

    # Enhanced Cellpose segmentation
    print("Running enhanced Cellpose segmentation...")
    model = models.CellposeModel(gpu=False, model_type='cyto')

    masks_pred = []
    for i, img in enumerate(imgs):
        print(f"Processing image {i+1}/{len(imgs)}")
        
        # Enhanced segmentation parameters
        mask, flows, styles = model.eval([img],
                                       diameter=None,
                                       channels=[0,0],
                                       flow_threshold=0.4,
                                       cellprob_threshold=0.0,
                                       min_size=15)
        
        if isinstance(mask, list):
            mask = mask[0]
        masks_pred.append(mask.astype(np.int32))

    # Save masks
    if masks_pred:
        masks_pred_path = os.path.join(folder_path, 'masks_pred_enhanced.npz')
        np.savez_compressed(masks_pred_path, masks=masks_pred)
        print(f"Enhanced masks saved to: {masks_pred_path}")

    # Extract comprehensive features
    print("Extracting comprehensive features...")
    all_features = []
    cell_data_records = []

    for frame_idx, (mask, img) in enumerate(zip(masks_pred, imgs)):
        if mask.shape != img.shape:
            print(f"Shape mismatch in frame {frame_idx}")
            continue

        features = extract_comprehensive_features(mask, img)
        all_features.append(features)

        # Store detailed records
        for cell_idx, feature_row in enumerate(features):
            record = create_comprehensive_cell_record(frame_idx, cell_idx, feature_row)
            cell_data_records.append(record)

        print(f"Frame {frame_idx}: {len(features)} cells detected")

    # Save comprehensive cell data
    if cell_data_records:
        cell_df = pd.DataFrame(cell_data_records)
        csv_path = os.path.join(folder_path, 'comprehensive_cell_data.csv')
        cell_df.to_csv(csv_path, index=False)
        print(f"Comprehensive cell data saved to: {csv_path}")

    # Create enhanced training dataset
    dataset = []
    total_pos_edges = 0
    total_neg_edges = 0

    for i in range(len(all_features) - 1):
        features_t1 = all_features[i]
        features_t2 = all_features[i + 1]

        if len(features_t1) == 0 or len(features_t2) == 0:
            continue

        combined_features = np.vstack([features_t1, features_t2])
        edge_index, labels = create_adaptive_temporal_graph(features_t1, features_t2)

        if edge_index.shape[1] == 0:
            continue
        
        num_positive_edges = len(labels)
        desired_negatives = int(num_positive_edges * (1 - negative_ratio) / negative_ratio)
        num_negative_edges = desired_negatives

        # Ensure minimum positive ratio
        max_negatives = int(num_positive_edges / min_positive_ratio - num_positive_edges)
        num_negative_edges = min(num_negative_edges, max_negatives)

        # Add negative sampling for better training balance
        num_nodes = len(combined_features)
        neg_edge_index = negative_sampling(
            torch.LongTensor(edge_index), 
            num_nodes=num_nodes,
            num_neg_samples=num_negative_edges
        )
        
        # Combine positive and negative edges
        full_edge_index = torch.cat([torch.LongTensor(edge_index), neg_edge_index], dim=1)
        full_labels = torch.cat([torch.FloatTensor(labels), torch.zeros(neg_edge_index.shape[1])])

        data = Data(
            x=torch.FloatTensor(combined_features),
            edge_index=full_edge_index,
            y=full_labels
        )
        dataset.append(data)

        total_pos_edges += num_positive_edges
        total_neg_edges += num_negative_edges

    print(f"Created {len(dataset)} enhanced temporal graphs with negative sampling")
    return dataset, all_features
