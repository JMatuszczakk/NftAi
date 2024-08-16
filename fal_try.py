import fal_client
import json
import random

# open accesories.json file that is a list
with open("accesories.json") as f:
    accesories = json.load(f)

# get 3 random accesories
accesories = random.sample(accesories, 3)


handler = fal_client.submit(
    "fal-ai/flux/schnell",
    arguments={
        "prompt": "A cryptopunk pixelart   nft of a tiger that has " + ", ".join(accesories) + ". Style is pixelart cryptopunk, so only the upper part of their body is visible, it has a tiger pattern, and it has a punk style. The background is homogeneous. The tiger is looking at the viewer with a fierce expression. It has a " + ", ".join(accesories) + ".",
    },
)
result = handler.get()

print(result)