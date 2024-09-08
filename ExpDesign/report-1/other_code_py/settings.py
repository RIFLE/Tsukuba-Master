import os
import pathlib

PRJ_DIR = pathlib.Path(__file__).parent.resolve()

TARGET_RESOLUTION = (512, 512) # (384, 384)  # (224, 224)
TARGET_RESOLUTION_IMAGE = TARGET_RESOLUTION + (3,)
TARGET_RESOLUTION_MODEL = TARGET_RESOLUTION + (1,)

PREFIX_IMAGES = r"Images"
PREFIX_ANNOTATIONS = r"annotations"

EXT_IMAGES = r".jpg"
EXT_ANNOTATIONS = r".png"

SPLIT_TRAIN_PERCENTAGE = 0.85 # 0.7
SPLIT_VALID_PERCENTAGE = SPLIT_TRAIN_PERCENTAGE + 0.1

SAVED_MODEL_FILENAME = r"HUGO-25_9500-500-0.85_1B_unet_video_segment_03.hdf5"

# SAVED_MODEL_FILENAME = r"HUGO-25_9500-500-0.85_unet_video_segment_05.hdf5"
# SAVED_MODEL_FILENAME = r"HUGO-25_500-0.85_BATCH1_unet_video_segment_03.hdf5"

# SAVED_MODEL_FILENAME = r"HUGO-25_600-0.85_containers_BATCH1_unet_video_segment_02.hdf5"

# SAVED_MODEL_FILENAME = r"HUGO-25_1900-100-0.85_unet_video_segment_04.hdf5"

# SAVED_MODEL_FILENAME = r"HUGO_exp_200_unet_video_segment_first.hdf5"
# SAVED_MODEL_FILENAME = r"WOW-25_450-50_unet_video_segment.hdf5"
# SAVED_MODEL_FILENAME = r"STOCH_HUGO-25_450-50-0.85_unet_video_segment.hdf5"



# SAVED_MODEL_FILENAME = r"1_unet_video_segment_random_set_first_test.hdf5"

BATCH_SIZE = 1   # 4
EPOCHS = 10  # 5

IS_PREPROCESSING_TRIGGER = False
NUM_AUGMENTED = 20  # 25

IMAGE_RESCALE_RATIO = 1.0 # 1.5

DB_NAME_TRAIN = r"db_train.npz"
DB_NAME_VALIDATE = r"db_validate.npz"
DB_NAME_TEST = r"db_test.npz"


#########################################################################################################
# DB_DATA = os.path.join(PRJ_DIR, r"db/data/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/labels/")


# training data set
#################################################################################

# DB_DATA = os.path.join(PRJ_DIR, r"db/WOW_10_10000/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/WOW_10_10000/training/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_10_10000/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_10_10000/training/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/training/images_containers/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/training/masks_containers/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/WOW_25_10000/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/WOW_25_10000/training/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_10000/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_10000/training/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/training/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/2000_HUGO_25/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/2000_HUGO_25/training/masks/")


# DB_DATA = os.path.join(PRJ_DIR, r"db/WOW_10_exp/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/WOW_10_exp/training/masks/")


# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/training/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/WOW_25_exp/training/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/WOW_25_exp/training/masks/")


# testing data set
####################################################################################

DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_10000/testing/images/")
DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_10000/testing/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/WOW_10_10000/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/WOW_10_10000/testing/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_10_10000/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_10_10000/testing/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/WOW_25_10000/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/WOW_25_10000/testing/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/testing/images_containers/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/testing/masks_containers/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/testing/images_containers/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/testing/masks_containers/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/testing/masks/")


# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_10000/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_10000/testing/masks/")


# DB_DATA = os.path.join(PRJ_DIR, r"db/2000_HUGO_25/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/2000_HUGO_25/testing/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/WOW_10_exp/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/WOW_10_exp/testing/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/WOW_25_exp/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/WOW_25_exp/testing/masks/")

# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_exp/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_exp/testing/masks/")


# DB_DATA = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/testing/images/")
# DB_LABEL = os.path.join(PRJ_DIR, r"db/HUGO_25_exp/testing/masks/")


#########################################################################################################