import requests
import os

def upload_image(server_url, image_path):
    # Ensure the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return

    # Prepare the file for upload
    with open(image_path, 'rb') as image_file:
        files = {'image': (os.path.basename(image_path), image_file, 'image/jpeg')}
        
        try:
            # Send POST request to the server
            response = requests.post(f"{server_url}/upload", files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                print("Image uploaded successfully!")
                print(response.json())
            else:
                print(f"Failed to upload image. Status code: {response.status_code}")
                print(response.text)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while uploading the image: {e}")

if __name__ == "__main__":
    # Replace with your server's URL
    SERVER_URL = "http://localhost:5000"
    
    # Replace with the path to your image file
    IMAGE_PATH = "path/to/your/image.jpg"
    
    upload_image(SERVER_URL, IMAGE_PATH)