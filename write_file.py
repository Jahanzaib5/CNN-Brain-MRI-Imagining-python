import os
from PIL import Image

path = 'normal/'
# Store the image file names in a list as long as they are jpgs
images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpeg']

with open("Output.txt", "w") as text_file:
    for image in images:
        text_file.write(image)
        text_file.write('\n')
