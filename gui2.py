import fal_client
import json
import random
import requests
from PIL import Image, ImageDraw
import io
import os
import math
import numpy as np
from skimage.feature import hog
from pathlib import Path
from scipy.spatial.distance import cosine
import cv2

# Create directories to save the images
os.makedirs("saved_images", exist_ok=True)
os.makedirs("grouped_images/group_4", exist_ok=True)

# Open accessories.json file that is a list
with open("accesories.json") as f:
    all_accessories = json.load(f)

def color_distance(color1, color2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

def check_color_similarity(img):
    width, height = img.size
    center_bottom = img.getpixel((width // 2, height - 1))
    bottom_right = img.getpixel((width - 1, height - 1))
    
    similarity_threshold = 10
    
    return color_distance(center_bottom, bottom_right) < similarity_threshold

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
        print(f"No images found in the folder: {folder_path}")
        return None
    
    all_features = np.array([extract_features(path) for path in image_paths])
    folder_style = np.mean(all_features, axis=0)
    
    return folder_style

def compare_image_to_folder(target_image, folder_path):
    folder_style = get_folder_style(folder_path)
    
    if folder_style is None:
        return 0  # Return 0 similarity if the folder is empty
    
    target_features = extract_features(target_image)
    similarity = 1 - cosine(target_features, folder_style)
    
    return similarity

class ImageGenerator:
    def __init__(self):
        self.image_count = 0
        self.max_images = 1000
    
    def generate_and_save_image(self):
        self.image_count += 1
        print(f"Generating image {self.image_count}/{self.max_images}")
        
        accessories = random.sample(all_accessories, 3)
        prompt = (f"A cryptopunk pixelart nft of a tiger that has {', '.join(accessories)}. "
                  f"Style is pixelart cryptopunk, so only the upper part of their body is visible, "
                  f"it has a tiger pattern, and it has a punk style. The background is homogeneous. The portrait is visible to the bottom of the image. "
                  f"The tiger is looking at the viewer with a fierce expression. The image is like a portrait to the bottom of the artpiece, not an icon"
                  f"The tiger is facing directly towards the viewer in a frontal pose. Its head is held high, giving an impression of confidence or coolness. The shoulders are visible, suggesting an upright, almost human-like posture."
                  f"It has a {', '.join(accessories)}."
                )

        while True:
            handler = fal_client.submit(
                "fal-ai/flux/schnell",
                arguments={"prompt": prompt},
            )
            result = handler.get()

            image_url = result['images'][0]['url']
            
            # Download the image
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content))
            
            if not check_color_similarity(img):
                # Save the image if it meets the color condition
                rin = random.randint(1000, 9999)
                filename = f"saved_images/tiger_{rin}.png"
                img.save(filename)

                # Add the number of the image to the upper right corner
                draw = ImageDraw.Draw(img)
                draw.text((0, 0), str(rin), (255, 255, 255))
                
                img.save("latest_image.png")
                print(f"Image saved as {filename}")

                # Check similarity with group_4
                similarity = compare_image_to_folder("latest_image.png", "grouped_images/group_4")
                print(f"Similarity score: {similarity:.4f}")

                if similarity > 0.7:
                    group4_filename = f"grouped_images/group_4/tiger_{rin}.png"
                    img.save(group4_filename)
                    print(f"Image also saved to group_4 as {group4_filename}")

                break
            else:
                print(f"Rejected image URL: {image_url}")

    def run(self):
        while self.image_count < self.max_images:
            self.generate_and_save_image()
        print("All done!")

# Main application
if __name__ == "__main__":
    generator = ImageGenerator()
    generator.run()