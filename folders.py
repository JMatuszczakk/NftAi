import os
import numpy as np
import shutil
from PIL import Image
from scipy.cluster.vq import kmeans, vq
from concurrent.futures import ThreadPoolExecutor

class ImageGrouper:
    def __init__(self, input_dir, output_dir, n_clusters=5):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        self.image_paths = []
        self.features = []
        self.valid_image_paths = []

    def run(self):
        self._load_images()
        self._extract_features()
        if len(self.features) > 0:
            labels = self._cluster_images()
            self._copy_to_groups(labels)
            self._print_summary(labels)
        else:
            print("No valid images to process.")

    def _load_images(self):
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))

    def _extract_features(self):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._process_image, self.image_paths))
        self.features = []
        self.valid_image_paths = []
        for path, feature in zip(self.image_paths, results):
            if feature is not None:
                self.features.append(feature)
                self.valid_image_paths.append(path)

    def _process_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img = img.convert('L').resize((100, 100))
                img_array = np.array(img)
                lbp = self._calculate_lbp(img_array)
                return lbp.flatten().astype(np.float64)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    @staticmethod
    def _calculate_lbp(image, P=8, R=1):
        rows, cols = image.shape
        result = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(P):
            x = R * np.cos(2 * np.pi * i / P)
            y = R * np.sin(2 * np.pi * i / P)
            xp = int(round(x))
            yp = int(round(y))
            neighbors = np.roll(np.roll(image, shift=-yp, axis=0), shift=-xp, axis=1)
            result = np.bitwise_or(result, (neighbors > image).astype(np.uint8) << i)
        return result

    def _cluster_images(self):
        if len(self.features) < self.n_clusters:
            print(f"Warning: Number of valid images ({len(self.features)}) is less than the number of clusters ({self.n_clusters})")
            self.n_clusters = max(2, len(self.features))
        features_array = np.array(self.features)
        centroids, _ = kmeans(features_array, self.n_clusters)
        labels, _ = vq(features_array, centroids)
        return labels

    def _copy_to_groups(self, labels):
        for label, src_path in zip(labels, self.valid_image_paths):
            dst_dir = os.path.join(self.output_dir, f'group_{label}')
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_path, dst_dir)

    def _print_summary(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Images have been grouped into {self.n_clusters} clusters.")
        print(f"Grouped images can be found in: {self.output_dir}")
        for group, count in zip(unique, counts):
            print(f"Group {group}: {count} images")
        print(f"Total valid images processed: {len(self.valid_image_paths)}")
        print(f"Images skipped due to errors: {len(self.image_paths) - len(self.valid_image_paths)}")

if __name__ == "__main__":
    input_directory = "saved_images"
    output_directory = "grouped_images"
    n_clusters = 5

    grouper = ImageGrouper(input_directory, output_directory, n_clusters)
    grouper.run()