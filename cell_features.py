from skimage.measure import regionprops
import numpy as np

def extract_comprehensive_features(masks, image, min_area=15):
    """Extract comprehensive features with normalization and additional metrics"""
    if masks.max() == 0:
        return np.empty((0, 12))

    features_list = []
    
    try:
        props = regionprops(masks.astype(int), intensity_image=image.astype(float))

        for prop in props:
            if prop.area < min_area:
                continue

            # Basic features
            centroid_y, centroid_x = prop.centroid
            area = prop.area
            mean_intensity = prop.mean_intensity
            max_intensity = prop.max_intensity
            perimeter = prop.perimeter
            
            # Shape features
            eccentricity = prop.eccentricity
            solidity = prop.solidity
            extent = prop.extent  # Area ratio to bounding box
            
            # Aspect ratio from bounding box
            minr, minc, maxr, maxc = prop.bbox
            height = maxr - minr
            width = maxc - minc
            aspect_ratio = height / width if width > 0 else 1.0
            
            # Additional shape metrics
            equivalent_diameter = prop.equivalent_diameter
            major_axis_length = prop.major_axis_length
            minor_axis_length = prop.minor_axis_length
            
            # Compactness (circularity measure)
            compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 1.0

            features_list.append([
                centroid_x, centroid_y,  # 0, 1: position
                area,                    # 2: size
                mean_intensity,          # 3: intensity
                max_intensity,           # 4: intensity
                perimeter,               # 5: shape
                eccentricity,            # 6: shape
                solidity,                # 7: shape
                aspect_ratio,            # 8: shape
                equivalent_diameter,     # 9: size
                major_axis_length,       # 10: shape
                compactness              # 11: shape
            ])

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return np.empty((0, 12))

    if len(features_list) == 0:
        return np.empty((0, 12))

    features = np.array(features_list)
    
    # Normalize features for better training
    features_normalized = features.copy()
    
    # Normalize position features by image size
    if len(features) > 0:
        features_normalized[:, 0] /= image.shape[1]  # x normalized by width
        features_normalized[:, 1] /= image.shape[0]  # y normalized by height
    
    return features_normalized

