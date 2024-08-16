import fal_client
import json
import random
import requests
from PIL import Image
import io
import os

# Create a directory to save the images
os.makedirs("saved_images", exist_ok=True)

# Open accessories.json file that is a list
with open("accesories.json") as f:
    all_accessories = json.load(f)

def generate_and_show_image():
    # Get 3 random accessories
    accessories = random.sample(all_accessories, 3)

    prompt = (f"A cryptopunk pixelart nft of a tiger that has {', '.join(accessories)}. "
              f"Style is pixelart cryptopunk, so only the upper part of their body is visible, "
              f"it has a tiger pattern, and it has a punk style. The background is homogeneous. "
              f"The tiger is looking at the viewer with a fierce expression. "
              f"It has a {', '.join(accessories)}.")

    handler = fal_client.submit(
        "fal-ai/flux/schnell",
        arguments={"prompt": prompt},
    )
    result = handler.get()

    image_url = result['images'][0]['url']
    
    # Download the image
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    
    # Display the image
    img.show()
    
    # Ask for user input
    user_input = input("Do you want to save this image? (y/n): ").lower()
    
    if user_input == 'y':
        # Save the image
        filename = f"saved_images/tiger_{random.randint(1000, 9999)}.png"
        img.save(filename)
        print(f"Image saved as {filename}")
    else:
        print("Image discarded")

# Main loop
for i in range(1000):
    print(f"Generating image {i+1}/1000")
    generate_and_show_image()
    print()

print("All done!")