import requests
import json
import fal_client
from PIL import Image
import io
import os
import threading
import time

def get_prompt():
    response = requests.get("http://sf.matuszczak.org:5002/prompt")
    return response.json()['prompt']

def generate_and_upload_image(thread_id):
    while True:
        # Generate image
        prompt = get_prompt()
        handler = fal_client.submit(
            "fal-ai/flux/schnell",
            arguments={"prompt": prompt},
        )
        result = handler.get()

        image_url = result['images'][0]['url']
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))
        
        # Save image temporarily
        image_path = f"flux_{thread_id}.png"
        img.save(image_path)

        # Upload image
        upload_image(image_path)

        # Remove temporary file
        os.remove(image_path)

def upload_image(image_path, server_url='http://sf.matuszczak.org:5002'):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return

    with open(image_path, 'rb') as image_file:
        files = {'image': (os.path.basename(image_path), image_file, 'image/jpeg')}
        
        try:
            response = requests.post(f"{server_url}/upload", files=files)
            
            if response.status_code == 200:
                print(f"Image {image_path} uploaded successfully!")
                print(response.json())
            else:
                print(f"Failed to upload image {image_path}. Status code: {response.status_code}")
                print(response.text)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while uploading the image {image_path}: {e}")

def main():
    # Start 5 generator/uploader threads
    threads = []
    for i in range(20):
        thread = threading.Thread(target=generate_and_upload_image, args=(i,), daemon=True)
        thread.start()
        threads.append(thread)

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping the program...")

if __name__ == "__main__":
    main()