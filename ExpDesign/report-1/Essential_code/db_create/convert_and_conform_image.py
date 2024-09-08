import os
import cv2
import numpy as np
import sys
from pathlib import Path

def create_npz_from_pgm(input_directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".pgm"):
            img = cv2.imread(os.path.join(input_directory, filename), cv2.IMREAD_GRAYSCALE)
            # normalize data
            data_normalized = img.astype('float32') / 255.0
            # expand dimensions to have 3 channels
            data_expanded = np.repeat(data_normalized[..., np.newaxis], 3, axis=2)
            np.savez(os.path.join(output_directory, filename[:-4]), data_expanded)
'''
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_pgm_to_npz.py <input_directory> <output_directory>")
        sys.exit()
'''
#    input_directory = sys.argv[1]
#    output_directory = sys.argv[2]


# create_npz_from_pgm("HUGO_10_ALL", "conforming/HUGO_10_ALL")
# create_npz_from_pgm("HUGO_25_ALL", "conforming/HUGO_25_ALL")
# create_npz_from_pgm("WOW_10_ALL", "conforming/WOW_10_ALL")
# create_npz_from_pgm("WOW_25_ALL", "conforming/WOW_25_ALL")

# create_npz_from_pgm("../2000_NORMAL", "./2000_NORMAL")
# create_npz_from_pgm("../2000_HUGO-25", "./2000_HUGO-25")

# WARNING: Every batch of 10000 worth of 30+ GB

# create_npz_from_pgm("HUGO-25/10000_HUGO-25", "HUGO-25/conforming_10000/10000_HUGO-25")
# print("HUGO-25 10000 images DONE")

# create_npz_from_pgm("HUGO-10/10000_HUGO-10", "HUGO-10/conforming_10000/10000_HUGO-10")
# print("HUGO-10 10000 images DONE")

# create_npz_from_pgm("WOW-25/10000_WOW-25", "WOW-25/conforming_10000/10000_WOW-25")
# print("WOW-25 10000 images DONE")

# create_npz_from_pgm("WOW-10/10000_WOW-10", "WOW-10/conforming_10000/10000_WOW-10")
# print("WOW-10 10000 images DONE")

create_npz_from_pgm("10000_NORMAL", "10000_NORMAL_conforming")
print("10000 NORMAL images DONE")
