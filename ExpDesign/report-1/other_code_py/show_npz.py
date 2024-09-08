import numpy as np
import matplotlib.pyplot as plt

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_10000/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_10000/testing/masks/")

# Load npz files
#data1 = np.load('db/HUGO_25_10000/training/images/4499.npz')
#data2 = np.load('db/HUGO_25_10000/training/masks/4499.npz')

data1 = np.load('db/HUGO_10_10000/training/images/1.npz')
data2 = np.load('db/HUGO_10_10000/training/masks/1.npz')

# data1 = np.load('db/HUGO_10_10000/testing/images/9501.npz')
# data2 = np.load('db/HUGO_10_10000/testing/masks/9501.npz')


# The .npz file can contain multiple arrays,
# you can load an image array by its variable name.
# Replace 'arr_0' with the actual variable name in your .npz file.
img_array1 = data1['arr_0']
img_array2 = data2['arr_0']

# Create a figure with 2 subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot image on each subplot
axs[0].imshow(img_array1)
axs[0].set_title('Image')

axs[1].imshow(img_array2)
axs[1].set_title('Mask')

# Display images
plt.show()