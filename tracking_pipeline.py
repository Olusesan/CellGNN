import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from lineage_logic import LineageTracker
from graph_builder import create_adaptive_temporal_graph
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment

def create_lineage_track_record(track_id, frame_idx, cell_idx, features, lineage_tracker):
    """Helper function to create track records with lineage information"""
    return {
        'track_id': track_id,
        'frame': frame_idx,
        'cell_id': f"f{frame_idx}_c{cell_idx}",
        'lineage_id': lineage_tracker.track_to_lineage.get(track_id, -1),
        'generation': lineage_tracker.generation_map.get(track_id, 0),
        'x': features[0],
        'y': features[1],
        'area': features[2],
        'mean_intensity': features[3],
        'max_intensity': features[4],
        'perimeter': features[5],
        'eccentricity': features[6],
        'solidity': features[7],
        'aspect_ratio': features[8],
        'equivalent_diameter': features[9] if len(features) > 9 else 0,
        'major_axis_length': features[10] if len(features) > 10 else 0,
        'compactness': features[11] if len(features) > 11 else 0
    }

def create_lineage_summary(tracks_df, lineage_tracker):
    """Create summary of each lineage"""
    lineage_summaries = []
    
    for lineage_id in set(lineage_tracker.track_to_lineage.values()):
        lineage_tracks = [tid for tid, lid in lineage_tracker.track_to_lineage.items() if lid == lineage_id]
        lineage_data = tracks_df[tracks_df['track_id'].isin(lineage_tracks)]
        
        if len(lineage_data) == 0:
            continue
            
        # Calculate lineage statistics
        total_tracks = len(lineage_tracks)
        total_frames = lineage_data['frame'].nunique()
        first_frame = lineage_data['frame'].min()
        last_frame = lineage_data['frame'].max()
        duration = last_frame - first_frame + 1
        
        # Count divisions in this lineage
        divisions_in_lineage = sum(1 for event in lineage_tracker.division_events 
                                 if event['parent_track'] in lineage_tracks)
        
        # Generation statistics
        generations = [lineage_tracker.generation_map[tid] for tid in lineage_tracks]
        max_generation = max(generations) if generations else 0
        
        # Root track (generation 0)
        root_tracks = [tid for tid in lineage_tracks if lineage_tracker.generation_map[tid] == 0]
        root_track = root_tracks[0] if root_tracks else None
        
        summary = {
            'lineage_id': lineage_id,
            'root_track_id': root_track,
            'total_tracks': total_tracks,
            'total_divisions': divisions_in_lineage,
            'max_generation': max_generation,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'duration_frames': duration,
            'active_frames': total_frames,
            'avg_area': lineage_data['area'].mean(),
            'total_detections': len(lineage_data)
        }
        lineage_summaries.append(summary)
    
    return pd.DataFrame(lineage_summaries)

def generate_lineage_statistics(tracks_df, lineage_tracker):
    """Generate comprehensive lineage and division statistics"""
    unique_tracks = tracks_df['track_id'].nunique()
    total_detections = len(tracks_df)
    unique_lineages = len(set(lineage_tracker.track_to_lineage.values()))
    total_divisions = len(lineage_tracker.division_events)
    
    # Generation statistics
    generations = list(lineage_tracker.generation_map.values())
    max_generation = max(generations) if generations else 0
    avg_generation = np.mean(generations) if generations else 0
    
    # Division statistics
    if lineage_tracker.division_events:
        division_frames = [event['frame'] for event in lineage_tracker.division_events]
        division_confidences = [event['division_confidence'] for event in lineage_tracker.division_events]
        avg_division_confidence = np.mean(division_confidences)
        
        # Daughters per division
        daughters_per_division = [len(event['daughter_tracks']) for event in lineage_tracker.division_events]
        avg_daughters = np.mean(daughters_per_division)
    else:
        division_frames = []
        avg_division_confidence = 0
        avg_daughters = 0
    
    # Track length analysis
    track_lengths = tracks_df.groupby('track_id').size()
    
    # Lineage analysis
    lineage_sizes = defaultdict(int)
    lineage_divisions = defaultdict(int)
    for tid, lid in lineage_tracker.track_to_lineage.items():
        lineage_sizes[lid] += 1
    
    for event in lineage_tracker.division_events:
        parent_lineage = lineage_tracker.track_to_lineage[event['parent_track']]
        lineage_divisions[parent_lineage] += 1
    
    frames_spanned = tracks_df['frame'].nunique()
    
    stats = f"""Enhanced Cell Lineage Tracking Statistics
==========================================
BASIC TRACKING:
- Total unique tracks: {unique_tracks}
- Total detections: {total_detections}
- Frames processed: {frames_spanned}
- Average cells per frame: {total_detections/frames_spanned:.1f}

LINEAGE ANALYSIS:
- Total lineages: {unique_lineages}
- Average tracks per lineage: {unique_tracks/unique_lineages:.1f}
- Largest lineage: {max(lineage_sizes.values()) if lineage_sizes else 0} tracks
- Most proliferative lineage: {max(lineage_divisions.values()) if lineage_divisions else 0} divisions

DIVISION ANALYSIS:
- Total division events: {total_divisions}
- Division rate: {total_divisions/frames_spanned:.3f} divisions per frame
- Average division confidence: {avg_division_confidence:.3f}
- Average daughters per division: {avg_daughters:.1f}
- Frames with divisions: {len(set(division_frames))}

GENERATION ANALYSIS:
- Maximum generation: {max_generation}
- Average generation: {avg_generation:.2f}
- Generation 0 (root) cells: {sum(1 for g in generations if g == 0)}
- Generation 1+ cells: {sum(1 for g in generations if g > 0)}

TRACK QUALITY:
- Average track length: {track_lengths.mean():.2f} frames
- Tracks ≥ 10 frames: {(track_lengths >= 10).sum()} ({100*(track_lengths >= 10).sum()/unique_tracks:.1f}%)
- Tracks ≥ 20 frames: {(track_lengths >= 20).sum()} ({100*(track_lengths >= 20).sum()/unique_tracks:.1f}%)

PROLIFERATION METRICS:
- Lineages with divisions: {sum(1 for count in lineage_divisions.values() if count > 0)}
- Non-proliferating lineages: {sum(1 for count in lineage_divisions.values() if count == 0)}
- Most divisions in single lineage: {max(lineage_divisions.values()) if lineage_divisions else 0}
"""
    
    return stats

def create_assignment_matrix(features_previous, features_current, edge_index, connection_probs):
    """Create cost matrix for Hungarian algorithm"""
    cost_matrix = np.full((len(features_previous), len(features_current)), 1.0)

    for i, prob in enumerate(connection_probs):
        prev_idx = edge_index[0][i]
        curr_idx = edge_index[1][i] - len(features_previous)
        if prev_idx < len(features_previous) and curr_idx < len(features_current):
            cost_matrix[prev_idx, curr_idx] = 1.0 - prob

    return cost_matrix


def save_lineage_tracking_results(track_records, lineage_tracker, folder_path):
    """Save tracking results with comprehensive lineage analysis"""
    if not track_records:
        print("No tracking data to save")
        return None, None

    # Create main tracking DataFrame
    tracks_df = pd.DataFrame(track_records)
    
    # Save main tracking results
    tracks_csv_path = os.path.join(folder_path, 'lineage_tracking_results.csv')
    tracks_df.to_csv(tracks_csv_path, index=False)
    
    # Save division events
    divisions_df = pd.DataFrame(lineage_tracker.division_events)
    divisions_csv_path = os.path.join(folder_path, 'division_events.csv')
    divisions_df.to_csv(divisions_csv_path, index=False)
    
    # Create lineage summary
    lineage_summary = create_lineage_summary(tracks_df, lineage_tracker)
    summary_path = os.path.join(folder_path, 'lineage_summary.csv')
    lineage_summary.to_csv(summary_path, index=False)
    
    # Generate comprehensive statistics
    stats = generate_lineage_statistics(tracks_df, lineage_tracker)
    stats_path = os.path.join(folder_path, 'lineage_statistics.txt')
    
    with open(stats_path, 'w') as f:
        f.write(stats)
    
    print(f"Lineage tracking results saved to: {tracks_csv_path}")
    print(f"Division events saved to: {divisions_csv_path}")
    print(f"Lineage summary saved to: {summary_path}")
    print(f"Statistics saved to: {stats_path}")
    print("\nLineage Tracking Summary:")
    print(stats)
    
    return tracks_df, lineage_tracker

def enhanced_tracking_with_lineage(all_features, trained_model, folder_path, 
                                  max_distance=75, max_gap_frames=3):
    """Enhanced tracking with lineage tracking and division detection"""
    
    if folder_path is None:
        folder_path = '/tracks'

    os.makedirs(folder_path, exist_ok=True)

    if len(all_features) < 2:
        print("Need at least 2 frames for tracking")
        return None, None

    print("Starting enhanced tracking with lineage analysis...")

    # Initialize lineage tracker
    lineage_tracker = LineageTracker()
    
    track_records = []
    track_id_counter = 0
    active_tracks = {}
    terminated_tracks = {}

    # Initialize tracks for first frame
    for cell_idx in range(len(all_features[0])):
        track_id = track_id_counter
        track_id_counter += 1

        # Add to lineage system (root cells)
        lineage_tracker.add_track(track_id)

        active_tracks[track_id] = {
            'last_position': all_features[0][cell_idx, :2],
            'last_frame': 0,
            'last_cell_idx': cell_idx,
            'last_features': all_features[0][cell_idx],
            'history': [all_features[0][cell_idx]],
            'lineage_id': lineage_tracker.track_to_lineage[track_id],
            'generation': lineage_tracker.generation_map[track_id]
        }

        # Add record with lineage information
        record = create_lineage_track_record(track_id, 0, cell_idx, all_features[0][cell_idx], lineage_tracker)
        track_records.append(record)

    trained_model.eval()

    # Process subsequent frames
    for frame_idx in range(1, len(all_features)):
        features_current = all_features[frame_idx]
        
        if len(features_current) == 0:
            # Move active tracks to terminated
            for track_id, track_info in active_tracks.items():
                terminated_tracks[track_id] = track_info
            active_tracks.clear()
            continue

        # Standard tracking logic with lineage updates
        if len(active_tracks) == 0:
            # Create new tracks for all current cells
            for cell_idx in range(len(features_current)):
                track_id = track_id_counter
                track_id_counter += 1

                # Add as new lineage root
                lineage_tracker.add_track(track_id)

                active_tracks[track_id] = {
                    'last_position': features_current[cell_idx, :2],
                    'last_frame': frame_idx,
                    'last_cell_idx': cell_idx,
                    'last_features': features_current[cell_idx],
                    'history': [features_current[cell_idx]],
                    'lineage_id': lineage_tracker.track_to_lineage[track_id],
                    'generation': lineage_tracker.generation_map[track_id]
                }

                record = create_lineage_track_record(track_id, frame_idx, cell_idx, features_current[cell_idx], lineage_tracker)
                track_records.append(record)
            continue

        # Create tracking graph and perform assignment (same logic as before)
        active_track_ids = list(active_tracks.keys())
        features_previous = np.array([active_tracks[tid]['last_features'] 
                                    for tid in active_track_ids])

        combined_features = np.vstack([features_previous, features_current])
        edge_index, _ = create_adaptive_temporal_graph(
            features_previous, features_current, max_distance=max_distance
        )

        if edge_index.shape[1] == 0:
            # No connections - terminate and create new
            for track_id in active_track_ids:
                terminated_tracks[track_id] = active_tracks[track_id]
            active_tracks.clear()

            for cell_idx in range(len(features_current)):
                track_id = track_id_counter
                track_id_counter += 1
                lineage_tracker.add_track(track_id)
                
                active_tracks[track_id] = {
                    'last_position': features_current[cell_idx, :2],
                    'last_frame': frame_idx,
                    'last_cell_idx': cell_idx,
                    'last_features': features_current[cell_idx],
                    'history': [features_current[cell_idx]],
                    'lineage_id': lineage_tracker.track_to_lineage[track_id],
                    'generation': lineage_tracker.generation_map[track_id]
                }
                record = create_lineage_track_record(track_id, frame_idx, cell_idx, features_current[cell_idx], lineage_tracker)
                track_records.append(record)
            continue

        # Use GNN and Hungarian algorithm
        with torch.no_grad():
            data = Data(
                x=torch.FloatTensor(combined_features),
                edge_index=torch.LongTensor(edge_index)
            )
            connection_probs = torch.sigmoid(trained_model(data.x, data.edge_index)).numpy()

        cost_matrix = create_assignment_matrix(features_previous, features_current, 
                                             edge_index, connection_probs)
        prev_indices, curr_indices = linear_sum_assignment(cost_matrix)

        # Process assignments
        assigned_current = set()
        new_active_tracks = {}

        for prev_idx, curr_idx in zip(prev_indices, curr_indices):
            if cost_matrix[prev_idx, curr_idx] < 0.6:  # Good match threshold
                track_id = active_track_ids[prev_idx]
                assigned_current.add(curr_idx)

                # Update track with lineage info preserved
                new_active_tracks[track_id] = {
                    'last_position': features_current[curr_idx, :2],
                    'last_frame': frame_idx,
                    'last_cell_idx': curr_idx,
                    'last_features': features_current[curr_idx],
                    'history': active_tracks[track_id]['history'] + [features_current[curr_idx]],
                    'lineage_id': active_tracks[track_id]['lineage_id'],
                    'generation': active_tracks[track_id]['generation']
                }

                record = create_lineage_track_record(track_id, frame_idx, curr_idx, features_current[curr_idx], lineage_tracker)
                track_records.append(record)
            else:
                # Terminate track
                track_id = active_track_ids[prev_idx]
                terminated_tracks[track_id] = active_tracks[track_id]

        # Create new tracks for unassigned cells
        for cell_idx in range(len(features_current)):
            if cell_idx not in assigned_current:
                track_id = track_id_counter
                track_id_counter += 1

                lineage_tracker.add_track(track_id)

                new_active_tracks[track_id] = {
                    'last_position': features_current[cell_idx, :2],
                    'last_frame': frame_idx,
                    'last_cell_idx': cell_idx,
                    'last_features': features_current[cell_idx],
                    'history': [features_current[cell_idx]],
                    'lineage_id': lineage_tracker.track_to_lineage[track_id],
                    'generation': lineage_tracker.generation_map[track_id]
                }

                record = create_lineage_track_record(track_id, frame_idx, cell_idx, features_current[cell_idx], lineage_tracker)
                track_records.append(record)

        active_tracks = new_active_tracks

        # Division detection every few frames
        if frame_idx % 3 == 0 and len(track_records) > 0:
            temp_df = pd.DataFrame(track_records)
            divisions = lineage_tracker.detect_division(temp_df, frame_idx)
            
            if divisions:
                print(f"Frame {frame_idx}: Detected {len(divisions)} potential division(s)")
                lineage_tracker.process_divisions(divisions)
                
                # Update generation info for affected tracks
                for division in divisions:
                    for daughter_track in division['daughter_tracks']:
                        if daughter_track in new_active_tracks:
                            new_active_tracks[daughter_track]['generation'] = lineage_tracker.generation_map[daughter_track]

    # Save comprehensive results with lineage
    return save_lineage_tracking_results(track_records, lineage_tracker, folder_path)

