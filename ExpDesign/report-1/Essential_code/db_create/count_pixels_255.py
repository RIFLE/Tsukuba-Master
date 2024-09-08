from PIL import Image
import numpy as np

# Open the image file
img = Image.open('1.pgm')

# Convert the image data to an array
data = np.array(img)

# Count the number of occurrences of 255
count = np.sum(data == 255)

print("Number of pixels equal to 255: ", count)