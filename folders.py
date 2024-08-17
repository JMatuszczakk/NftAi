import os
import numpy as np
import shutil
from PIL import Image
from scipy.cluster.vq import kmeans, vq
from scipy.signal import convolve2d
from concurrent.futures import ThreadPoolExecutor

class ImageGrouper:
    def __init__(self, input_dir, output_dir, n_clusters=5):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        self.image_paths = []
        self.features = []

    def run(self):
        self._load_images()
        self._extract_features()
        labels = self._cluster_images()
        self._copy_to_groups(labels)
        self._print_summary(labels)

    def _load_images(self):
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))

    def _extract_features(self):
        with ThreadPoolExecutor() as executor:
            self.features = list(executor.map(self._process_image, self.image_paths))

    def _process_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert('L').resize((100, 100))
            img_array = np.array(img)
            return self._calculate_lbp(img_array).flatten()

    @staticmethod
    def _calculate_lbp(image, P=8, R=1):
        rows, cols = image.shape
        result = np.zeros_like(image)
        for i in range(P):
            x = R * np.cos(2 * np.pi * i / P)
            y = R * np.sin(2 * np.pi * i / P)
            xp = np.round(x).astype(int)
            yp = np.round(y).astype(int)
            neighbors = np.roll(np.roll(image, shift=-yp, axis=0), shift=-xp, axis=1)
            result += (neighbors > image) * (1 << i)
        return result

    def _cluster_images(self):
        centroids, _ = kmeans(np.array(self.features), self.n_clusters)
        labels, _ = vq(self.features, centroids)
        return labels

    def _copy_to_groups(self, labels):
        for label, src_path in zip(labels, self.image_paths):
            dst_dir = os.path.join(self.output_dir, f'group_{label}')
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_path, dst_dir)

    def _print_summary(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Images have been grouped into {self.n_clusters} clusters.")
        print(f"Grouped images can be found in: {self.output_dir}")
        for group, count in zip(unique, counts):
            print(f"Group {group}: {count} images")

if __name__ == "__main__":
    input_directory = "saved_images"
    output_directory = "folders"
    n_clusters = 10

    grouper = ImageGrouper(input_directory, output_directory, n_clusters)
    grouper.run()