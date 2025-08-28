import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
from skimage.measure import regionprops
from collections import defaultdict
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

def visualize_masks_and_tracks(tracks_df, images, masks, output_path=None, 
                              frame_range=None, max_tracks_to_show=15,
                              show_track_ids=True, show_boundaries=True,
                              colormap='viridis'):
    """
    Create comprehensive visualization showing masks and tracks overlaid
    
    Parameters:
    - tracks_df: DataFrame with tracking results
    - images: List of original images
    - masks: List of segmentation masks
    - output_path: Path to save visualization
    - frame_range: Tuple (start, end) frames to visualize, or None for all
    - max_tracks_to_show: Maximum number of tracks to display
    - show_track_ids: Whether to show track ID labels
    - show_boundaries: Whether to show cell boundaries
    - colormap: Colormap for tracks
    """
    if output_path is None:
        output_path = '/content/drive/MyDrive/masks_and_tracks_visualization.png'
    
    # Determine frame range
    if frame_range is None:
        frame_range = (0, min(len(images), len(masks), tracks_df['frame'].max() + 1))
    
    start_frame, end_frame = frame_range
    frames_to_show = min(6, end_frame - start_frame)  # Show up to 6 frames
    frame_indices = np.linspace(start_frame, end_frame - 1, frames_to_show, dtype=int)
    
    # Select diverse tracks to show
    track_lengths = tracks_df.groupby('track_id').size()
    track_activity = tracks_df.groupby('track_id')['frame'].agg(['min', 'max'])
    
    # Prioritize tracks that are active in our frame range
    active_tracks = []
    for track_id in track_lengths.index:
        track_start, track_end = track_activity.loc[track_id]
        if track_start <= end_frame - 1 and track_end >= start_frame:
            active_tracks.append(track_id)
    
    # Select a diverse set of tracks
    if len(active_tracks) > max_tracks_to_show:
        # Get longest tracks first, then sample others
        active_track_lengths = track_lengths[active_tracks]
        longest_tracks = active_track_lengths.nlargest(max_tracks_to_show // 2).index.tolist()
        remaining_tracks = [t for t in active_tracks if t not in longest_tracks]
        if remaining_tracks:
            np.random.seed(42)  # For reproducibility
            sampled_tracks = np.random.choice(
                remaining_tracks, 
                size=min(max_tracks_to_show - len(longest_tracks), len(remaining_tracks)),
                replace=False
            ).tolist()
            selected_tracks = longest_tracks + sampled_tracks
        else:
            selected_tracks = longest_tracks
    else:
        selected_tracks = active_tracks
    
    # Create visualization
    fig, axes = plt.subplots(2, frames_to_show, figsize=(4 * frames_to_show, 8))
    if frames_to_show == 1:
        axes = axes.reshape(-1, 1)
    
    # Color setup
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(selected_tracks)))
    track_colors = {track_id: colors[i] for i, track_id in enumerate(selected_tracks)}
    
    for col, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(images) or frame_idx >= len(masks):
            continue
            
        image = images[frame_idx]
        mask = masks[frame_idx]
        
        # Top row: Original image with masks overlay
        ax_top = axes[0, col]
        ax_top.imshow(image, cmap='gray', alpha=0.7)
        
        if mask.max() > 0:
            # Create colored mask overlay
            if show_boundaries:
                boundaries = find_boundaries(mask, mode='thick')
                mask_colored = label2rgb(mask, image, alpha=0.3, bg_label=0)
                ax_top.imshow(mask_colored)
                ax_top.imshow(boundaries, cmap='Reds', alpha=0.5)
            else:
                mask_colored = label2rgb(mask, alpha=0.4, bg_label=0)
                ax_top.imshow(mask_colored)
        
        ax_top.set_title(f'Frame {frame_idx}\nMasks + Boundaries')
        ax_top.axis('off')
        
        # Bottom row: Tracking results
        ax_bottom = axes[1, col]
        ax_bottom.imshow(image, cmap='gray', alpha=0.6)
        
        # Plot tracks up to this frame
        frame_tracks = tracks_df[
            (tracks_df['frame'] <= frame_idx) & 
            (tracks_df['track_id'].isin(selected_tracks))
        ]
        
        # Plot track trajectories
        for track_id in selected_tracks:
            track_data = frame_tracks[frame_tracks['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) > 1:
                ax_bottom.plot(track_data['x'], track_data['y'], 
                             color=track_colors[track_id], linewidth=2, alpha=0.8)
        
        # Plot current frame detections
        current_detections = tracks_df[
            (tracks_df['frame'] == frame_idx) & 
            (tracks_df['track_id'].isin(selected_tracks))
        ]
        
        for _, detection in current_detections.iterrows():
            color = track_colors[detection['track_id']]
            
            # Plot detection point
            ax_bottom.plot(detection['x'], detection['y'], 'o', 
                         color=color, markersize=8, markeredgecolor='white', 
                         markeredgewidth=1)
            
            # Add track ID label if requested
            if show_track_ids:
                ax_bottom.annotate(f"{int(detection['track_id'])}", 
                                 (detection['x'], detection['y']),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=8, fontweight='bold',
                                 color='white', 
                                 bbox=dict(boxstyle='round,pad=0.2', 
                                         facecolor=color, alpha=0.7))
        
        ax_bottom.set_title(f'Frame {frame_idx}\nTracks (n={len(current_detections)})')
        ax_bottom.axis('off')
    
    # Add overall title and legend
    fig.suptitle(f'Cell Segmentation and Tracking\n'
                f'Frames {start_frame}-{end_frame-1}, '
                f'{len(selected_tracks)} tracks shown', fontsize=14, fontweight='bold')
    
    # Create legend for tracks
    legend_elements = []
    for i, track_id in enumerate(selected_tracks[:10]):  # Limit legend entries
        legend_elements.append(plt.Line2D([0], [0], color=track_colors[track_id], 
                                        linewidth=3, label=f'Track {track_id}'))
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(legend_elements)))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Masks and tracks visualization saved to: {output_path}")
    plt.show()

def visualize_lineage_tree(tracks_df, lineage_tracker, output_path=None, max_lineages=10):
    """Create visualization of lineage trees"""
    if output_path is None:
        output_path = '/content/drive/MyDrive/lineage_trees.png'
    
    # Select most interesting lineages (with divisions)
    lineage_division_counts = defaultdict(int)
    for event in lineage_tracker.division_events:
        parent_lineage = lineage_tracker.track_to_lineage[event['parent_track']]
        lineage_division_counts[parent_lineage] += 1
    
    # Sort lineages by division count and size
    lineage_scores = {}
    for lineage_id in set(lineage_tracker.track_to_lineage.values()):
        tracks_in_lineage = [tid for tid, lid in lineage_tracker.track_to_lineage.items() if lid == lineage_id]
        divisions = lineage_division_counts[lineage_id]
        size = len(tracks_in_lineage)
        lineage_scores[lineage_id] = divisions * 2 + size  # Weight divisions more
    
    selected_lineages = sorted(lineage_scores.keys(), key=lambda x: lineage_scores[x], reverse=True)[:max_lineages]
    
    # Create subplot layout
    n_lineages = len(selected_lineages)
    cols = min(3, n_lineages)
    rows = (n_lineages + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_lineages == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    for idx, lineage_id in enumerate(selected_lineages):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break
            
        # Get tracks in this lineage
        tracks_in_lineage = [tid for tid, lid in lineage_tracker.track_to_lineage.items() if lid == lineage_id]
        lineage_data = tracks_df[tracks_df['track_id'].isin(tracks_in_lineage)]
        
        if len(lineage_data) == 0:
            ax.set_title(f'Lineage {lineage_id} (Empty)')
            continue
        
        # Create a simple tree layout
        generations = {}
        for track_id in tracks_in_lineage:
            gen = lineage_tracker.generation_map[track_id]
            if gen not in generations:
                generations[gen] = []
            generations[gen].append(track_id)
        
        # Plot tracks by generation
        colors = plt.cm.Set3(np.linspace(0, 1, len(tracks_in_lineage)))
        track_colors = {track_id: colors[i] for i, track_id in enumerate(tracks_in_lineage)}
        
        for gen, tracks in generations.items():
            for i, track_id in enumerate(tracks):
                track_data = lineage_data[lineage_data['track_id'] == track_id].sort_values('frame')
                
                # Plot trajectory with generation offset
                y_offset = gen * 50 + i * 10
                ax.plot(track_data['frame'], track_data['x'] + y_offset, 
                       color=track_colors[track_id], linewidth=2, alpha=0.8,
                       label=f'T{track_id} G{gen}' if len(tracks_in_lineage) <= 8 else None)
                
                # Mark start and end
                ax.plot(track_data['frame'].iloc[0], track_data['x'].iloc[0] + y_offset, 
                       'o', color=track_colors[track_id], markersize=6)
                ax.plot(track_data['frame'].iloc[-1], track_data['x'].iloc[-1] + y_offset, 
                       's', color=track_colors[track_id], markersize=6)
        
        # Mark division events
        for event in lineage_tracker.division_events:
            if event['parent_track'] in tracks_in_lineage:
                parent_gen = lineage_tracker.generation_map[event['parent_track']]
                ax.axvline(x=event['frame'], color='red', linestyle='--', alpha=0.7)
        
        divisions_count = lineage_division_counts[lineage_id]
        ax.set_title(f'Lineage {lineage_id}\n{len(tracks_in_lineage)} tracks, {divisions_count} divisions')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Position + Generation Offset')
        
        if len(tracks_in_lineage) <= 8:
            ax.legend(fontsize=8, loc='upper left')
    
    # Hide unused subplots
    for idx in range(n_lineages, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Lineage tree visualization saved to: {output_path}")
    plt.show()

def visualize_tracking_results(tracks_df, output_path=None, max_tracks_to_show=20):
    """Create visualization of tracking results"""
    if output_path is None:
        output_path = '/content/drive/MyDrive/tracking_visualization.png'
    
    plt.figure(figsize=(15, 10))
    
    # Get track lengths and select diverse tracks to show
    track_lengths = tracks_df.groupby('track_id').size()
    
    # Select tracks: longest tracks + some medium + some short
    long_tracks = track_lengths.nlargest(max_tracks_to_show//2).index
    medium_tracks = track_lengths.iloc[len(track_lengths)//3:len(track_lengths)*2//3].sample(
        min(max_tracks_to_show//4, len(track_lengths)//3), random_state=42
    ).index
    short_tracks = track_lengths.nsmallest(max_tracks_to_show//4).index
    
    selected_tracks = list(long_tracks) + list(medium_tracks) + list(short_tracks)
    selected_tracks = selected_tracks[:max_tracks_to_show]
    
    # Plot selected tracks
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_tracks)))
    
    for i, track_id in enumerate(selected_tracks):
        track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) > 1:  # Only plot tracks with multiple points
            plt.plot(track_data['x'], track_data['y'], 
                    color=colors[i], linewidth=2, alpha=0.7, 
                    label=f'Track {track_id} ({len(track_data)} pts)')
            
            # Mark start and end
            plt.plot(track_data['x'].iloc[0], track_data['y'].iloc[0], 
                    'o', color=colors[i], markersize=8, alpha=0.9)
            plt.plot(track_data['x'].iloc[-1], track_data['y'].iloc[-1], 
                    's', color=colors[i], markersize=8, alpha=0.9)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Cell Tracking Results - {len(selected_tracks)} Selected Tracks\n'
              f'○ = Start, □ = End')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tracking visualization saved to: {output_path}")
    plt.show()

def visualize_division_analysis(tracks_df, lineage_tracker, output_path=None):
    """Create comprehensive division analysis visualizations"""
    if output_path is None:
        output_path = '/content/drive/MyDrive/division_analysis.png'
    
    if not lineage_tracker.division_events:
        print("No division events detected for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Division events over time
    division_frames = [event['frame'] for event in lineage_tracker.division_events]
    axes[0,0].hist(division_frames, bins=max(10, len(set(division_frames))//2), alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Division Events Over Time')
    axes[0,0].set_xlabel('Frame')
    axes[0,0].set_ylabel('Number of Divisions')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Division confidence distribution
    confidences = [event['division_confidence'] for event in lineage_tracker.division_events]
    axes[0,1].hist(confidences, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[0,1].set_title('Division Confidence Distribution')
    axes[0,1].set_xlabel('Confidence Score')
    axes[0,1].set_ylabel('Number of Divisions')
    axes[0,1].axvline(x=np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Generation distribution
    generations = list(lineage_tracker.generation_map.values())
    generation_counts = pd.Series(generations).value_counts().sort_index()
    axes[1,0].bar(generation_counts.index, generation_counts.values, alpha=0.7, color='green')
    axes[1,0].set_title('Cell Generation Distribution')
    axes[1,0].set_xlabel('Generation')
    axes[1,0].set_ylabel('Number of Cells')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Lineage size vs divisions
    lineage_sizes = defaultdict(int)
    lineage_divisions = defaultdict(int)
    
    for tid, lid in lineage_tracker.track_to_lineage.items():
        lineage_sizes[lid] += 1
    
    for event in lineage_tracker.division_events:
        parent_lineage = lineage_tracker.track_to_lineage[event['parent_track']]
        lineage_divisions[parent_lineage] += 1
    
    sizes = [lineage_sizes[lid] for lid in lineage_sizes.keys()]
    divisions = [lineage_divisions[lid] for lid in lineage_sizes.keys()]
    
    axes[1,1].scatter(sizes, divisions, alpha=0.6, s=50, color='purple')
    axes[1,1].set_title('Lineage Size vs Division Count')
    axes[1,1].set_xlabel('Number of Tracks in Lineage')
    axes[1,1].set_ylabel('Number of Divisions')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(sizes) > 1:
        correlation = np.corrcoef(sizes, divisions)[0,1]
        axes[1,1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                      transform=axes[1,1].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Division analysis saved to: {output_path}")
    plt.show()
