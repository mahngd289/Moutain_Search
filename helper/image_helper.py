from PIL import Image
import os


def resize_images(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(input_folder, filename))
            img = img.resize(size, Image.LANCZOS)
            img.save(os.path.join(output_folder, filename))


resize_images("raw_images", "processed_images")