import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import hog
from skimage import io, color, transform

def load_images(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(directory, filename)
            img = io.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def preprocess_image(image):
    # Resize image to a standard size
    image = transform.resize(image, (128, 128))
    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = color.rgb2gray(image)
    return image

def extract_features(image):
    # Extract HOG features
    features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=True)
    return features

def cluster_images(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

def group_images(directory, n_clusters=5):
    # Load images
    images, filenames = load_images(directory)
    
    # Preprocess images and extract features
    features = []
    for image in images:
        preprocessed = preprocess_image(image)
        features.append(extract_features(preprocessed))
    
    # Convert features to numpy array
    features = np.array(features)
    
    # Cluster images
    labels = cluster_images(features, n_clusters)
    
    # Group images by cluster
    groups = {}
    for filename, label in zip(filenames, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(filename)
    
    return groups

