import cv2
import numpy as np
from skimage.feature import hog
from pathlib import Path
from scipy.spatial.distance import cosine

def extract_features(image_path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    features, _ = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def get_folder_style(folder_path):
    folder_path = Path(folder_path)
    image_paths = [p for p in folder_path.glob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_paths:
        raise ValueError(f"No images found in the folder: {folder_path}")
    
    # Extract features for all images in the folder
    all_features = np.array([extract_features(path) for path in image_paths])
    
    # Compute the mean feature vector to represent the folder's style
    folder_style = np.mean(all_features, axis=0)
    
    return folder_style

def compare_image_to_folder(target_image, folder_path):
    target_path = Path(target_image)
    folder_path = Path(folder_path)

    # Check if paths exist
    if not target_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_image}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Folder path is not a directory: {folder_path}")

    # Get the folder's style
    folder_style = get_folder_style(folder_path)
    
    # Extract features from the target image
    target_features = extract_features(target_path)
    
    # Compute similarity using cosine similarity
    similarity = 1 - cosine(target_features, folder_style)
    
    return similarity

def main():
    target_image = "latest_image.png"
    folder_path = "grouped_images/group_4"
    
    try:
        similarity = compare_image_to_folder(target_image, folder_path)
        print(f"\nSimilarity of '{target_image}' to the style of images in '{folder_path}':")
        print(f"Similarity score: {similarity:.4f}")
        
        # Interpret the similarity score
        if similarity > 0.8:
            print("The image is very similar to the folder's style.")
        elif similarity > 0.6:
            print("The image is moderately similar to the folder's style.")
        else:
            print("The image is not very similar to the folder's style.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except NotADirectoryError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()