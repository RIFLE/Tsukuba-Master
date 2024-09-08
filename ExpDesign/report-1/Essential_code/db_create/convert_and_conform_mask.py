import os
import cv2
import numpy as np
import sys
from pathlib import Path

def create_npz_from_pgm_mask(input_directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".pgm"):
            img = cv2.imread(os.path.join(input_directory, filename), cv2.IMREAD_GRAYSCALE)
            # convert dtype to float32 with normalization
            img_float32 = img.astype('float32') / 255.0 # precalculated mask values defined as {0, 255} so normalize
            np.savez(os.path.join(output_directory, filename[:-4]), img_float32)
'''
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_pgm_to_npz_mask.py <input_directory> <output_directory>")
        sys.exit()
'''

# input_directory = sys.argv[1]
# output_directory = sys.argv[2]

# create_npz_from_pgm_mask("../Masks/512/1/HUGO-masks-1_512_100/", "conforming/HUGO-masks_512_100/")

# create_npz_from_pgm_mask("MASK-HUGO_10_ALL", "conforming/MASK-HUGO_10_ALL")
# create_npz_from_pgm_mask("MASK-HUGO_25_ALL", "conforming/MASK-HUGO_25_ALL")
# create_npz_from_pgm_mask("MASK-WOW_10_ALL", "conforming/MASK-WOW_10_ALL")
# create_npz_from_pgm_mask("MASK-WOW_25_ALL", "conforming/MASK-WOW_25_ALL")
# create_npz_from_pgm_mask("empty_container/empty_mask", "conforming/empty_mask")
# create_npz_from_pgm_mask("../masks-2000_hugo_25", "./2000_MASKS-HUGO-25")

create_npz_from_pgm_mask("HUGO-25/10000_MASKS-HUGO-25", "HUGO-25/conforming_10000/10000_MASKS-HUGO-25")
print("HUGO-25 10000 masks DONE")

create_npz_from_pgm_mask("HUGO-10/10000_MASKS-HUGO-10", "HUGO-10/conforming_10000/10000_MASKS-HUGO-10")
print("HUGO-10 10000 masks DONE")

create_npz_from_pgm_mask("WOW-25/10000_MASKS-WOW-25", "WOW-25/conforming_10000/10000_MASKS-WOW-25")
print("WOW-25 10000 masks DONE")

create_npz_from_pgm_mask("WOW-10/10000_MASKS-WOW-10", "WOW-10/conforming_10000/10000_MASKS-WOW-10")
print("WOW-10 10000 masks DONE")
