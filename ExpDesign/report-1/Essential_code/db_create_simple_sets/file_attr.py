import numpy as np


def print_npz_info(npz_path):
    npz_file = np.load(npz_path)

    for key, value in npz_file.items():
        print(f"Array '{key}':")
        print(f"- dtype: {value.dtype}")
        print(f"- shape: {value.shape}")
        print(f"- min: {np.min(value)}")
        print(f"- max: {np.max(value)}")
        print("\n")
'''
print("Original file")
print_npz_info("data/1.npz")

print("Created embedded data file")
print_npz_info("HUGO_train_100/HUGO_512_100/5.npz")
'''

print("Conforming file")
# print_npz_info("conforming_2000/2000_NORMAL/201.npz")
# print_npz_info("conforming_2000/2000_HUGO-25/201.npz")
print_npz_info("HUGO_25_10000/training/images/3.npz")

print_npz_info("HUGO_25_10000/training/images/9500.npz")

print_npz_info("HUGO_25_10000/testing/images/9501.npz")

print_npz_info("HUGO_25_10000/testing/images/10000.npz")





'''
print("Original mask")
print_npz_info("labels/25.npz")

print("Created mask")
print_npz_info("HUGO_train_100/HUGO-masks-1_512_100/1.npz")
'''
print("Conforming mask")

print_npz_info("HUGO_25_10000/training/masks/3.npz")

print_npz_info("HUGO_25_10000/training/masks/9500.npz")

print_npz_info("HUGO_25_10000/testing/masks/9501.npz")

print_npz_info("HUGO_25_10000/testing/masks/10000.npz")

# print_npz_info("conforming_2000/empty_mask_2000/201.npz")
# print_npz_info("conforming_2000/2000_MASKS-HUGO-25/201.npz")