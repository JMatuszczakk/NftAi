#@title Generate images

collection_name = "dooggies" #@param {type:"string"}

nrows = 1 #@param {type:"integer"}

generation_type = "default" #@param ["default", "ema"]

from huggingnft.lightweight_gan.train import timestamped_filename

from huggingnft.lightweight_gan.lightweight_gan import load_lightweight_model

from IPython.display import Image



model = load_lightweight_model(f"huggingnft/{collection_name}")

image_saved_path, generated_image = model.generate_app(

    num=timestamped_filename(),

    nrow=nrows,

    checkpoint=-1,

    types=generation_type,

)


for i in range(100):
    image_saved_path, generated_image = model.generate_app(

    num=timestamped_filename(),

    nrow=nrows,

    checkpoint=-1,

    types=generation_type,)
    print("done"+str(i))
    print(image_saved_path)

Image(image_saved_path)
