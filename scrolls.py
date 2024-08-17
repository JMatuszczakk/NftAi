
import fal_client
import json
import random

# open accesories.json file that is a list
with open("scrolls.json") as f:
    accesories = json.load(f)

# get 3 random accesories
accesories = random.sample(accesories, 1)


handler = fal_client.submit(
    "fal-ai/flux/schnell",
    arguments={
        "prompt": f"create a pixel art cryptopunk kind of an image of a {str(accesories[0])}, the background should be teal. The text on the scroll is japanese. The scroll is old and worn out.",
    },
)
result = handler.get()

print(result)