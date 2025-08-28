
from tracking_pipeline import enhanced_tracking_with_lineage
import os
import numpy as np
from visualization import visualize_lineage_tree, visualize_tracking_results, visualize_division_analysis

def analyze_tracking_quality(tracks_df):
    """Analyze and report tracking quality metrics"""
    print("\nDetailed Tracking Quality Analysis ")
    
    # Basic statistics
    unique_tracks = tracks_df['track_id'].nunique()
    total_detections = len(tracks_df)
    frames = tracks_df['frame'].nunique()
    
    print(f"Dataset Overview:")
    print(f"Total tracks: {unique_tracks}")
    print(f"Total detections: {total_detections}")
    print(f"Frames: {frames}")
    print(f"Average detections per frame: {total_detections/frames:.1f}")
    
    # Track length analysis
    track_lengths = tracks_df.groupby('track_id').size()
    
    print(f"\nTrack Length Distribution:")
    print(f"Mean: {track_lengths.mean():.2f} frames")
    print(f"Median: {track_lengths.median():.2f} frames")
    print(f"Std: {track_lengths.std():.2f} frames")
    print(f"Min: {track_lengths.min()} frames")
    print(f"Max: {track_lengths.max()} frames")
    
    # Quality categories
    very_short = (track_lengths == 1).sum()
    short = ((track_lengths >= 2) & (track_lengths < 5)).sum()
    medium = ((track_lengths >= 5) & (track_lengths < 15)).sum()
    long_tracks = ((track_lengths >= 15) & (track_lengths < 30)).sum()
    very_long = (track_lengths >= 30).sum()
    
    print(f"\nTrack Quality Categories:")
    print(f"Very short (1 frame): {very_short} ({100*very_short/unique_tracks:.1f}%)")
    print(f"Short (2-4 frames): {short} ({100*short/unique_tracks:.1f}%)")
    print(f"Medium (5-14 frames): {medium} ({100*medium/unique_tracks:.1f}%)")
    print(f"Long (15-29 frames): {long_tracks} ({100*long_tracks/unique_tracks:.1f}%)")
    print(f"Very long (≥30 frames): {very_long} ({100*very_long/unique_tracks:.1f}%)")
    
    # Movement analysis
    if 'x' in tracks_df.columns and 'y' in tracks_df.columns:
        movements = []
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            if len(track_data) > 1:
                distances = np.sqrt(np.diff(track_data['x'])**2 + np.diff(track_data['y'])**2)
                movements.extend(distances)
        
        if movements:
            movements = np.array(movements)
            print(f"\n Movement Analysis:")
            print(f"Mean step size: {movements.mean():.2f} pixels")
            print(f"Median step size: {np.median(movements):.2f} pixels")
            print(f"Max step size: {movements.max():.2f} pixels")
    
    # Temporal coverage
    frame_coverage = tracks_df.groupby('frame').size()
    print(f"\nTemporal Coverage:")
    print(f"Frames with detections: {len(frame_coverage)}/{frames}")
    print(f"Average cells per frame: {frame_coverage.mean():.1f}")
    print(f"Frame with most cells: {frame_coverage.max()}")
    print(f"Frame with fewest cells: {frame_coverage.min()}")

def run_enhanced_lineage_pipeline(all_features, folder_path=None, trained_model=None, max_frames=50,
                                   visualize=True, use_pretrained=True, force_retrain=False):
    """Run the complete enhanced cell tracking pipeline with lineage analysis"""

    print("Starting Enhanced Cell Tracking & Lineage Pipeline")

    try:
        # Step 1: Run enhanced tracking with lineage
        result = enhanced_tracking_with_lineage(
            all_features=all_features,
            trained_model=trained_model,
            folder_path=folder_path
        )

        if result is None or result[0] is None:
            print("Pipeline failed")
            return None, None, None

        tracks_df, lineage_tracker = result

        # Step 2: Quality analysis
        analyze_tracking_quality(tracks_df)

        # Step 3: Lineage-specific analysis
        print("\n Performing lineage analysis...")
        lineage_stats = lineage_tracker.get_lineage_statistics()
        print(f"Detected {lineage_stats['total_lineages']} lineages with {lineage_stats['total_divisions']} divisions")

        # Step 4: Visualizations
        if visualize:
            print("\n Generating visualizations...")
            output_base_path = folder_path or '/content/drive/MyDrive'

            # Standard tracking visualization
            visualize_tracking_results(
                tracks_df,
                output_path=os.path.join(output_base_path, 'tracking_results.png')
            )

            # Lineage tree visualization
            visualize_lineage_tree(
                tracks_df,
                lineage_tracker,
                output_path=os.path.join(output_base_path, 'lineage_trees.png')
            )

            # Division analysis
            if lineage_tracker.division_events:
                visualize_division_analysis(
                    tracks_df,
                    lineage_tracker,
                    output_path=os.path.join(output_base_path, 'division_analysis.png')
                )

        # # Step 5: Export for external tools
        # print("\n Exporting data for external analysis tools...")
        # export_lineage_for_external_tools(tracks_df, lineage_tracker, output_base_path)

        # print("\n Enhanced Cell Tracking & Lineage Pipeline Complete!")
        # print("=" * 60)

        return tracks_df, lineage_tracker, trained_model

    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None
        
    
