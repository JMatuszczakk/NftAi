import fal_client
import json
import random
import requests
from PIL import Image, ImageDraw
import io
import os
import math


# Create a directory to save the images
os.makedirs("saved_images", exist_ok=True)

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

                # to the upper right corner add the number of the image
                draw = ImageDraw.Draw(img)
                draw.text((0, 0), str(rin), (255, 255, 255))
                
           


                img.save("latest_image.png")
                print(f"Image saved as {filename}")
                break
            else:
                print(f"Rejected image URL: {image_url}")

    def run(self):
        while self.image_count < self.max_images:
            self.generate_and_save_image()
        print("All done!")

# Main application
generator = ImageGenerator()
generator.run()