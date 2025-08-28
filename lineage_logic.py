import networkx as nx
import numpy as np
from collections import defaultdict

class LineageTracker:
    """Class to handle cell lineage tracking and division detection"""
    
    def __init__(self):
        self.lineage_tree = nx.DiGraph()  # Directed graph for lineage
        self.track_to_lineage = {}        # Map track_id to lineage_id
        self.lineage_counter = 0          # Counter for unique lineages
        self.division_events = []         # List of division events
        self.generation_map = {}          # Track generation numbers
        
    def add_track(self, track_id, parent_track=None):
        """Add a new track to the lineage system"""
        if parent_track is None:
            # Root cell (no parent)
            lineage_id = self.lineage_counter
            self.lineage_counter += 1
            self.track_to_lineage[track_id] = lineage_id
            self.lineage_tree.add_node(track_id, lineage_id=lineage_id, generation=0)
            self.generation_map[track_id] = 0
        else:
            # Child cell
            parent_lineage = self.track_to_lineage[parent_track]
            parent_generation = self.generation_map[parent_track]
            
            self.track_to_lineage[track_id] = parent_lineage
            self.generation_map[track_id] = parent_generation + 1
            
            self.lineage_tree.add_node(track_id, 
                                     lineage_id=parent_lineage, 
                                     generation=parent_generation + 1)
            self.lineage_tree.add_edge(parent_track, track_id)
    
    def detect_division(self, tracks_df, frame_idx, area_threshold=1.5, 
                       distance_threshold=30, time_window=3):
        """Detect cell division events based on area increase and spatial proximity"""
        divisions = []
        
        # Get tracks that are active in the current frame and previous frames
        current_frame_tracks = tracks_df[tracks_df['frame'] == frame_idx]['track_id'].unique()
        
        for track_id in current_frame_tracks:
            # Get track history
            track_history = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_history) < time_window:
                continue
                
            # Check for area increase pattern (potential pre-division)
            recent_areas = track_history['area'].tail(time_window).values
            if len(recent_areas) >= 2:
                area_ratio = recent_areas[-1] / recent_areas[0]
                
                if area_ratio > area_threshold:
                    # Look for potential daughter cells
                    parent_pos = track_history.iloc[-1][['x', 'y']].values
                    
                    # Find nearby tracks that started recently
                    potential_daughters = []
                    for other_track in current_frame_tracks:
                        if other_track == track_id:
                            continue
                            
                        other_history = tracks_df[tracks_df['track_id'] == other_track].sort_values('frame')
                        
                        # Check if this track started recently (within time window)
                        if (frame_idx - other_history['frame'].min()) <= time_window:
                            other_pos = other_history.iloc[-1][['x', 'y']].values
                            distance = np.linalg.norm(parent_pos - other_pos)
                            
                            if distance <= distance_threshold:
                                potential_daughters.append(other_track)
                    
                    # If we found potential daughters, record division
                    if len(potential_daughters) >= 1:
                        division_event = {
                            'frame': frame_idx,
                            'parent_track': track_id,
                            'daughter_tracks': potential_daughters,
                            'parent_area': recent_areas[-1],
                            'division_confidence': min(1.0, area_ratio / area_threshold)
                        }
                        divisions.append(division_event)
        
        return divisions
    
    def process_divisions(self, division_events):
        """Process detected divisions and update lineage tree"""
        for event in division_events:
            parent_track = event['parent_track']
            daughter_tracks = event['daughter_tracks']
            
            # Add division event to records
            self.division_events.append(event)
            
            # Update lineage tree
            for daughter_track in daughter_tracks:
                if daughter_track not in self.track_to_lineage:
                    self.add_track(daughter_track, parent_track)
    
    def get_lineage_statistics(self):
        """Calculate lineage statistics"""
        stats = {
            'total_lineages': len(set(self.track_to_lineage.values())),
            'total_divisions': len(self.division_events),
            'max_generation': max(self.generation_map.values()) if self.generation_map else 0,
            'tracks_per_lineage': defaultdict(int),
            'generations_per_lineage': defaultdict(list)
        }
        
        for track_id, lineage_id in self.track_to_lineage.items():
            stats['tracks_per_lineage'][lineage_id] += 1
            stats['generations_per_lineage'][lineage_id].append(self.generation_map[track_id])
        
        return stats