from PIL import Image
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert JPG images to PNG')
parser.add_argument('--dir_path', type=str, required=True, help='Directory containing the images')
args = parser.parse_args()

dir_path = args.dir_path

# Loop over all JPG files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.jpg'):
        # Open the image file
        img = Image.open(os.path.join(dir_path, filename))
        # Get the base name of the file (without extension)
        base_name = os.path.splitext(filename)[0]
        # Save the image as PNG
        img.save(os.path.join(dir_path, base_name + '.png'))
