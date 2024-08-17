import cv2
import numpy as np
from skimage.feature import hog
from pathlib import Path
from scipy.spatial.distance import cdist

def extract_features(image_path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    features, _ = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def batch_extract_features(image_paths):
    return np.array([extract_features(path) for path in image_paths])

def compare_images(target_image, folder_path):
    target_path = Path(target_image)
    folder_path = Path(folder_path)

    # Check if paths exist
    if not target_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_image}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Folder path is not a directory: {folder_path}")

    # Get all image paths
    image_paths = [p for p in folder_path.glob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_paths:
        print(f"No images found in the folder: {folder_path}")
        return []

    # Extract features for all images including the target
    all_features = batch_extract_features([target_path] + image_paths)
    
    # Separate target features and folder images features
    target_features = all_features[0]
    folder_features = all_features[1:]
    
    # Compute distances
    distances = cdist([target_features], folder_features, metric='cosine')[0]
    
    # Convert distances to similarities
    similarities = 1 - distances
    
    # Create a list of (image_name, similarity) tuples
    results = list(zip([p.name for p in image_paths], similarities))
    
    # Sort by similarity (highest to lowest)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def main():
    target_image = "latest_image.png"
    folder_path = "grouped_images/group_4"
    
    try:
        results = compare_images(target_image, folder_path)
        
        if results:
            print("\nSimilarity results:")
            for name, similarity in results:
                print(f"{name}: {similarity:.2f}")
        else:
            print("No results to display.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except NotADirectoryError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()