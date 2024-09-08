import os

###
import shutil
from datetime import datetime
###

from copy import deepcopy

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import rescale, resize

from settings import BATCH_SIZE, DB_NAME_TEST, DB_NAME_TRAIN, DB_NAME_VALIDATE, EPOCHS, EXT_ANNOTATIONS, EXT_IMAGES, \
    IMAGE_RESCALE_RATIO, IS_PREPROCESSING_TRIGGER, NUM_AUGMENTED, PREFIX_ANNOTATIONS, PREFIX_IMAGES, PRJ_DIR, \
    SAVED_MODEL_FILENAME, SPLIT_TRAIN_PERCENTAGE, SPLIT_VALID_PERCENTAGE, TARGET_RESOLUTION, TARGET_RESOLUTION_IMAGE, \
    DB_DATA, DB_LABEL
from utils import init_model, list2tensor
from db_generator import DBGenerator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# try run on CPU only
# Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# Segmentation Models: using `keras` framework.
from tensorflow import keras
keras.backend.set_image_data_format('channels_last')   # channels_first

import albumentations as A

# create transformer for augmentation
transformer_albumentation = A.Compose([
    A.RandomCrop(width=TARGET_RESOLUTION_IMAGE[0], height=TARGET_RESOLUTION_IMAGE[1]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# prepare model
model_unet, model_preprocess_input = init_model()
is_color = True

# Ideas
# https://github.com/qubvel/segmentation_models
# https://albumentations.ai/
# https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset/
# https://arxiv.org/pdf/2003.02899.pdf
# https://github.com/Alex-Mathai-98/Satellite-Image-Segmentation-Using-U-NET
# https://github.com/robmarkcole/satellite-image-deep-learning
# https://ai.facebook.com/blog/using-ai-to-detect-covid-19-misinformation-and-exploitative-content/


def split2pieces(matrix, scaler_rows: int = 2, scaler_cols: int = 2, target_resolution=TARGET_RESOLUTION,
                 is_mask: bool = False):
    rows = np.arange(matrix.shape[0])
    cols = np.arange(matrix.shape[1])

    rows_parts = np.array_split(rows, indices_or_sections=scaler_rows)
    cols_parts = np.array_split(cols, indices_or_sections=scaler_cols)

    out = list()
    for curr_row_range in rows_parts:
        for curr_col_range in cols_parts:
            part = matrix[curr_row_range, :][:, curr_col_range]
            part_rescaled = resize(part, target_resolution, anti_aliasing=True, preserve_range=True).astype(part.dtype)

            if is_mask:
                part_rescaled = rgb2gray(part_rescaled)
                part_rescaled[part_rescaled > 0] = 1.0

            out.append(part_rescaled)
    return out


def get_dataset(is_verbose: bool = True, is_augment: bool = True, rows_splitter: int = 2, cols_slitter: int = 2,
                is_color: bool = False):
    # get list of filenames
    filename_list = os.listdir(os.path.join(PRJ_DIR, PREFIX_IMAGES))
    filename_list = [os.path.splitext(curr_file)[0] for curr_file in filename_list]

    db_raw = list()
    db_segment = list()

    target_resolution = TARGET_RESOLUTION
    if is_color:
        target_resolution = TARGET_RESOLUTION_IMAGE

    # iterate over image/annotation
    for curr_filename in filename_list:
        curr_filepath_image = os.path.join(PRJ_DIR, PREFIX_IMAGES, curr_filename + EXT_IMAGES)
        curr_filepath_annotation = os.path.join(PRJ_DIR, PREFIX_ANNOTATIONS, curr_filename + EXT_ANNOTATIONS)

        # load test image
        image = imread(curr_filepath_image)

        # load annotation
        image_mask = imread(curr_filepath_annotation)

        # rescale both image and annotation to same resolution
        image_rescaled = resize(image, TARGET_RESOLUTION_IMAGE, anti_aliasing=True)
        image_mask_rescaled = resize(image_mask, TARGET_RESOLUTION_IMAGE, anti_aliasing=True)

        # convert images to grayscale
        image_prepared = image_rescaled.copy()
        if not is_color:
            image_prepared = rgb2gray(image_rescaled)
        image_mask_prepared = rgb2gray(image_mask_rescaled)
        image_mask_prepared[image_mask_prepared > 0] = 1.0

        # augment data
        if is_augment:
            image_prepared = split2pieces(image_prepared,
                                          scaler_rows=rows_splitter,
                                          scaler_cols=cols_slitter,
                                          target_resolution=target_resolution)
            image_mask_prepared = split2pieces(image_mask_prepared,
                                               scaler_rows=rows_splitter,
                                               scaler_cols=cols_slitter,
                                               target_resolution=target_resolution)

            db_raw.extend(image_prepared)
            db_segment.extend(image_mask_prepared)
        else:
            # pack prepared images for further training with U-net
            db_raw.append(image_prepared)
            db_segment.append(image_mask_prepared)

        if is_verbose:
            print("File {} is processed".format(curr_filename))

    return db_raw, db_segment


def get_dataset_v2(transformer, is_verbose: bool = True, is_color: bool = False, num_augmented: int = 0,
                   rows_splitter: int = 2, cols_slitter: int = 2):
    def prepare_image_mask(image, image_mask, target_resolution):
        # rescale both image and annotation to same resolution
        image_rescaled = resize(image, target_resolution, anti_aliasing=True)
        image_mask_rescaled = resize(image_mask, target_resolution, anti_aliasing=True)

        # convert images to grayscale
        image_prepared = image_rescaled.copy()
        if not is_color:
            image_prepared = rgb2gray(image_rescaled)
        image_mask_prepared = rgb2gray(image_mask_rescaled)
        image_mask_prepared[image_mask_prepared > 0] = 1.0

        return image_prepared, image_mask_prepared

    # get list of filenames
    filename_list = os.listdir(os.path.join(PRJ_DIR, PREFIX_IMAGES))
    filename_list = [os.path.splitext(curr_file)[0] for curr_file in filename_list]

    db_raw = list()
    db_segment = list()

    target_resolution = TARGET_RESOLUTION
    if is_color:
        target_resolution = TARGET_RESOLUTION_IMAGE

    # iterate over image/annotation
    for curr_filename in filename_list:
        curr_filepath_image = os.path.join(PRJ_DIR, PREFIX_IMAGES, curr_filename + EXT_IMAGES)
        curr_filepath_annotation = os.path.join(PRJ_DIR, PREFIX_ANNOTATIONS, curr_filename + EXT_ANNOTATIONS)

        # load test image
        image = imread(curr_filepath_image)

        # load annotation
        image_mask = imread(curr_filepath_annotation)

        # append reference
        image_prepared, image_mask_prepared = prepare_image_mask(image, image_mask, target_resolution)

        # pack data
        db_raw.append(image_prepared)
        db_segment.append(image_mask_prepared)

        # initial augmentation
        image_prepared_aug = split2pieces(image_prepared, scaler_rows=rows_splitter, scaler_cols=cols_slitter,
                                          target_resolution=target_resolution)
        image_mask_prepared_aug = split2pieces(image_mask_prepared, scaler_rows=rows_splitter, scaler_cols=cols_slitter,
                                               target_resolution=target_resolution, is_mask=True)

        db_raw.extend(image_prepared_aug)
        db_segment.extend(image_mask_prepared_aug)

        # start augmentation
        # reference: https://albumentations.ai/docs/getting_started/mask_augmentation/
        if num_augmented > 0:
            for _ in np.arange(num_augmented):
                transformed = transformer(image=image, mask=image_mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']

                image_prepared, image_mask_prepared = prepare_image_mask(transformed_image, transformed_mask, target_resolution)

                # pack data
                db_raw.append(image_prepared)
                db_segment.append(image_mask_prepared)

        if is_verbose:
            print("File {} is processed".format(curr_filename))

    return db_raw, db_segment


def augment_data(db_init: np.array, db_segment: np.array):
    sample_number = db_init.shape[0]

    db_init_augmented = list()
    db_segment_augmented = list()
    for idx in np.arange(sample_number):
        if is_color:
            curr_image = db_init[idx, ...]
        else:
            curr_image = db_init[idx, :, :, 0]
        curr_segment = db_segment[idx, :, :, 0]

        # rotate data
        for idx_rotation in np.arange(4):
            rotate_image = np.rot90(curr_image, idx_rotation)
            rotate_segment = np.rot90(curr_segment, idx_rotation)

            for idx_flip in np.arange(3):
                flipped_image = rotate_image.copy()
                flipped_segment = rotate_segment.copy()
                if idx_flip != 2:
                    # flip image
                    flipped_image = np.flip(flipped_image, axis=idx_flip)
                    flipped_segment = np.flip(flipped_segment, axis=idx_flip)

                db_init_augmented.append(flipped_image)
                db_segment_augmented.append(flipped_segment)

    # restore initial shape (idx, height, width, color_depth)
    db_init_augmented = list2tensor(db_init_augmented, is_color=is_color)
    db_segment_augmented = list2tensor(db_segment_augmented, is_color=is_color)

    return db_init_augmented, db_segment_augmented


def sample_preprocessing(sample_list: list):
    from skimage.exposure import equalize_adapthist

    sample_processed = deepcopy(sample_list)

    for idx, curr_sample in enumerate(sample_list):
        sample_processed[idx] = equalize_adapthist(curr_sample)

    return sample_processed


def load_image_db(is_verbose: bool = True):
    # get list of filenames
    filename_list = os.listdir(os.path.join(PRJ_DIR, PREFIX_IMAGES))
    filename_list = [os.path.splitext(curr_file)[0] for curr_file in filename_list]

    db_raw = list()
    db_segment = list()

    target_resolution = TARGET_RESOLUTION
    if is_color:
        target_resolution = TARGET_RESOLUTION_IMAGE

    # iterate over image/annotation
    for curr_filename in filename_list:
        curr_filepath_image = os.path.join(PRJ_DIR, PREFIX_IMAGES, curr_filename + EXT_IMAGES)
        curr_filepath_annotation = os.path.join(PRJ_DIR, PREFIX_ANNOTATIONS, curr_filename + EXT_ANNOTATIONS)

        # load test image
        image = imread(curr_filepath_image)

        # load annotation
        image_mask = imread(curr_filepath_annotation)

        # pack data
        db_raw.append(image)
        db_segment.append(image_mask)

        if is_verbose:
            print("File {} is processed".format(curr_filename))

    return db_raw, db_segment


def augment_data_v2(image: np.array, mask: np.array, triggers: dict, opts: dict, is_color: bool = True):
    def prepare_image_mask(_image, _image_mask, target_resolution):
        # rescale both image and annotation to same resolution
        image_rescaled = resize(_image, target_resolution, anti_aliasing=True) #, preserve_range=True).astype(image.dtype)
        image_mask_rescaled = resize(_image_mask, target_resolution, anti_aliasing=True)

        # convert images to grayscale
        image_prepared = image_rescaled.copy()
        if not is_color:
            image_prepared = rgb2gray(image_rescaled)
        image_mask_prepared = rgb2gray(image_mask_rescaled)
        image_mask_prepared[image_mask_prepared > 0] = 1.0

        return image_prepared, image_mask_prepared

    def affine_augmentation(_image, _image_mask):
        image_list, mask_list = list(), list()
        for idx_rotation in np.arange(4):
            rotate_image = np.rot90(_image, idx_rotation)
            rotate_segment = np.rot90(_image_mask, idx_rotation)

            for idx_flip in np.arange(3):
                flipped_image = rotate_image.copy()
                flipped_segment = rotate_segment.copy()
                if idx_flip != 2:
                    # flip image
                    flipped_image = np.flip(flipped_image, axis=idx_flip)
                    flipped_segment = np.flip(flipped_segment, axis=idx_flip)

                image_list.append(flipped_image)
                mask_list.append(flipped_segment)

        return image_list, mask_list

    image_processed = list()
    mask_processed = list()

    target_resolution = TARGET_RESOLUTION
    if is_color:
        target_resolution = TARGET_RESOLUTION_IMAGE

    # prevent cropping for small image
    image = rescale(image, scale=IMAGE_RESCALE_RATIO, anti_aliasing=True, multichannel=True).astype(np.float32)
    mask = rescale(mask, scale=IMAGE_RESCALE_RATIO, anti_aliasing=True, multichannel=True).astype(np.float32)

    # append reference
    image_prepared, image_mask_prepared = prepare_image_mask(image, mask, target_resolution)

    # pack data
    image_processed.append(image_prepared)
    mask_processed.append(image_mask_prepared)

    try:
        # parse options
        # split image into parts
        if triggers['image_splitting']:
            image_prepared_aug = split2pieces(image,
                                              scaler_rows=opts['rows_splitter'],
                                              scaler_cols=opts['cols_slitter'],
                                              target_resolution=target_resolution)
            image_mask_prepared_aug = split2pieces(mask,
                                                   scaler_rows=opts['rows_splitter'],
                                                   scaler_cols=opts['cols_slitter'],
                                                   target_resolution=target_resolution,
                                                   is_mask=True)

            for idx_image in np.arange(image_prepared_aug.__len__()):
                image_processed_temp, mask_processed_temp = prepare_image_mask(image_prepared_aug[idx_image],
                                                                               image_mask_prepared_aug[idx_image],
                                                                               target_resolution)

                # rotation-flipping augmentation
                if triggers['image_affine']:
                    image_list, mask_list = affine_augmentation(image_processed_temp, mask_processed_temp)
                    image_processed.extend(image_list)
                    mask_processed.extend(mask_list)
                else:
                    image_processed.append(image_processed_temp)
                    mask_processed.append(mask_processed_temp)

        # albumentations-lib features
        if triggers['image_albumentation']:
            # reference: https://albumentations.ai/docs/getting_started/mask_augmentation/
            transformer = opts['transformer']
            if opts['num_augmented'] > 0:
                for _ in np.arange(opts['num_augmented']):
                    transformed = transformer(image=image, mask=mask)
                    transformed_image = transformed['image']
                    transformed_mask = transformed['mask']

                    image_prepared, image_mask_prepared = prepare_image_mask(transformed_image,
                                                                             transformed_mask,
                                                                             target_resolution)

                    # rotation-flipping augmentation
                    if triggers['image_affine']:
                        image_list, mask_list = affine_augmentation(image_prepared, image_mask_prepared)
                        image_processed.extend(image_list)
                        mask_processed.extend(mask_list)
                    else:
                        # pack data
                        image_processed.append(image_prepared)
                        mask_processed.append(image_mask_prepared)

    except Exception as exc:
        raise Exception('Error. Something went wrong during data augmentation - {}'.format(str(exc)))

    return image_processed, mask_processed


def augment_data_v2_wrapper(image_list: list, mask_list: list, triggers: dict, opts: dict, is_color: bool = True):
    out_image = list()
    out_mask = list()

    if image_list.__len__() != mask_list.__len__():
        raise ValueError('Error. Lists of images and corresponding segmentation masks have different size')

    for idx in np.arange(image_list.__len__()):
        image_processed, mask_processed = augment_data_v2(image_list[idx],
                                                          mask_list[idx],
                                                          triggers=triggers,
                                                          opts=opts,
                                                          is_color=is_color)
        out_image.extend(image_processed)
        out_mask.extend(mask_processed)

    return out_image, out_mask


def shuffle_db(image_list, mask_list):
    if image_list.__len__() != mask_list.__len__():
        raise ValueError('Error. Lists of images and corresponding segmentation masks have different size')

    image_number = image_list.__len__()
    sample_range = np.arange(image_number)
    sample_range_shuffled = np.random.permutation(sample_range)
    db_image = [image_list[idx] for idx in sample_range_shuffled]
    db_segmented = [mask_list[idx] for idx in sample_range_shuffled]

    return db_image, db_segmented


def part_process_wrapper(image_list, mask_list, augment_opt: dict):
    db_init, db_segment = augment_data_v2_wrapper(image_list, mask_list, **augment_opt)

    # shuffle dataset
    db_init, db_segment = shuffle_db(db_init, db_segment)

    # convert to tensors
    # db_init = list2tensor(db_init, is_color=is_color)
    # db_segment = list2tensor(db_segment, is_color=False)

    # Warning! preprocess input image for compatibility with pretrained model
    # db_init = model_preprocess_input(db_init)

    return db_init, db_segment


def prepare_db():
    # source: https://github.com/zhixuhao/unet

    # Refactored
    # set options
    augmentation_triggers = {
        'image_splitting': True,
        'image_albumentation': True,
        'image_affine': True,
    }
    augmentation_opts = {
        'rows_splitter': 4,
        'cols_slitter': 4,
        'num_augmented': NUM_AUGMENTED,
        'transformer': transformer_albumentation,
    }
    augment_opt = {
        'triggers': augmentation_triggers,
        'opts': augmentation_opts,
        'is_color': is_color,
    }

    # load database
    db_image, db_segmented = load_image_db()

    # shuffle set
    image_number = db_image.__len__()
    db_image, db_segmented = shuffle_db(db_image, db_segmented)

    # save db
    db_image_aug, db_segmented_aug = part_process_wrapper(db_image, db_segmented, augment_opt)
    for idx in np.arange(db_image_aug.__len__()):
        curr_x = db_image_aug[idx]
        curr_label = db_segmented_aug[idx]

        np.savez(os.path.join(PRJ_DIR, r"db/data/", str(idx) + '.npz'), curr_x)
        np.savez(os.path.join(PRJ_DIR, r"db/labels/", str(idx) + '.npz'), curr_label)


def train_wrapper(filename_data, filename_label):

    if filename_data.__len__() != filename_label.__len__():
        raise ValueError('Error. Sizes of data and label arrays are different')

    image_number = filename_data.__len__()

    # shuffle data
    filename_data_pre, filename_label_pre = shuffle_db(filename_data, filename_label)

    # split into train-validate-test
    image_split_train = int(SPLIT_TRAIN_PERCENTAGE * image_number)
    image_split_valid = int(SPLIT_VALID_PERCENTAGE * image_number)
    db_init_train, db_init_valid, db_init_test = filename_data_pre[:image_split_train], \
                                                 filename_data_pre[image_split_train:image_split_valid], \
                                                 filename_data_pre[image_split_valid:]
    db_segmented_train, db_segmented_valid, db_segmented_test = filename_label_pre[:image_split_train], \
                                                                filename_label_pre[image_split_train:image_split_valid], \
                                                                filename_label_pre[image_split_valid:]

    # prepare generators
    generator_train = DBGenerator(db_init_train, db_segmented_train, BATCH_SIZE)
    generator_valid = DBGenerator(db_init_valid, db_segmented_valid, BATCH_SIZE)
    generator_test = DBGenerator(db_init_test, db_segmented_test, BATCH_SIZE)

    # convert prepared data into appropriate tensorflow format for computations speed up
    # BATCH_SIZE_DATASET = 4
    # BATCH_SIZE_TRAIN = 4  # 32
    # BATCH_SIZE_VALID = 4  # 40
    # BUFFER_SIZE = 25
    # SPLIT_SIZE_DATASET = 2500  # 2500 / 5000
    #
    # train_dataset = tf.data.Dataset.from_tensor_slices((db_init_train.astype(np.float16), db_segmented_train.astype(np.float16)))
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE_DATASET)
    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    #
    # valid_dataset = tf.data.Dataset.from_tensor_slices((db_init_valid.astype(np.float16), db_segmented_valid.astype(np.float16)))
    # valid_dataset = valid_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE_DATASET)
    # valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    #
    # test_dataset = tf.data.Dataset.from_tensor_slices((db_init_test.astype(np.float16), db_segmented_test.astype(np.float16)))
    # test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE_DATASET)
    # test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    #####################
    #
    # # prepare dataset
    # db_init, db_segmented = get_dataset_v2(transformer=transformer_albumentation, is_color=is_color, num_augmented=NUM_AUGMENTED,
    #                                        rows_splitter=4, cols_slitter=4)
    # # db_init, db_segmented = get_dataset(is_color=is_color)
    #
    # # perform some preprocessing
    # if IS_PREPROCESSING_TRIGGER:
    #     db_init_ref = deepcopy(db_init)
    #     db_init = sample_preprocessing(db_init)
    #
    # # convert to tensors
    # db_init = list2tensor(db_init, is_color=is_color)
    # db_segmented = list2tensor(db_segmented, is_color=False)
    #
    # # shuffle dataset
    # sample_range = np.arange(db_init.shape[0])
    # sample_range_shuffled = np.random.permutation(sample_range)
    # db_init = db_init[sample_range_shuffled, ...]
    # db_segmented = db_segmented[sample_range_shuffled, ...]
    #
    # # split dataset
    # image_number = db_init.shape[0]
    # image_split_train = int(SPLIT_TRAIN_PERCENTAGE * image_number)
    # image_split_valid = int(SPLIT_VALID_PERCENTAGE * image_number)
    # db_init_train, db_init_valid, db_init_test = db_init[:image_split_train, ...], db_init[
    #                                                                                image_split_train:image_split_valid,
    #                                                                                ...], db_init[image_split_valid:,
    #                                                                                      ...]
    # db_segmented_train, db_segmented_valid, db_segmented_test = db_segmented[:image_split_train, ...], db_segmented[
    #                                                                                                    image_split_train:image_split_valid,
    #                                                                                                    ...], db_segmented[
    #                                                                                                          image_split_valid:,
    #                                                                                                          ...]
    #
    # # augment train data
    # db_init_train, db_segmented_train = augment_data(db_init_train, db_segmented_train)
    # db_init_valid, db_segmented_valid = augment_data(db_init_valid, db_segmented_valid)
    # db_init_test, db_segmented_test = augment_data(db_init_test, db_segmented_test)

    ############################

    print('Number of data in: train - {}, validation - {}, test - {}'.format(db_init_train.__len__(),
                                                                             db_init_valid.__len__(),
                                                                             db_init_test.__len__()))

    # train model
    # with tf.device('/CPU:0'):
    model_checkpoint = ModelCheckpoint(SAVED_MODEL_FILENAME[:-5] + '_{epoch:02d}.hdf5', monitor='loss', verbose=1,
                                       save_best_only=False)
    # model_checkpoint = ModelCheckpoint(SAVED_MODEL_FILENAME, monitor='loss', verbose=1, save_best_only=True)
    model_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, mode='auto', verbose=1)
    model_unet.fit(generator_train,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   validation_data=generator_valid,
                   callbacks=[model_checkpoint,
                              model_early_stop, ],
                   shuffle=True)
    # model_unet.fit(db_init_train, db_segmented_train,
    #                batch_size=BATCH_SIZE,
    #                epochs=EPOCHS,
    #                validation_data=(db_init_valid, db_segmented_valid),
    #                callbacks=[model_checkpoint, ])
    # model_unet.fit(train_dataset,
    #                batch_size=BATCH_SIZE,
    #                epochs=EPOCHS,
    #                validation_data=valid_dataset,
    #                validation_batch_size=BATCH_SIZE_VALID,
    #                shuffle=True,
    #                callbacks=[model_checkpoint, ])

    # validate data and estimate accuracy
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model_unet.evaluate(generator_test, batch_size=BATCH_SIZE)
    # results = model_unet.evaluate(test_dataset, batch_size=BATCH_SIZE)
    print("test loss, test acc:", results)


def evaluate_wrapper(filename_data, filename_label):
    if filename_data.__len__() != filename_label.__len__():
        raise ValueError('Error. Sizes of data and label arrays are different')

    print('Number of data in evaluation - {}'.format(filename_data.__len__()))

    # prepare generators
    generator_evaluate = DBGenerator(filename_data, filename_label, BATCH_SIZE)

    #BACKUP#BACKUP#BACKUP#BACKUP#BACKUP#BACKUP#BACKUP#

    #backup_file = f"{os.path.join(PRJ_DIR, SAVED_MODEL_FILENAME)}_{datetime.now().strftime('%Y%m%d%H%M%S')}.backup.hdf5"
    #shutil.copy2(SAVED_MODEL_FILENAME,backup_file)
    #print(f"Backup of {SAVED_MODEL_FILENAME} saved as {backup_file}")

    # prepare model
    model_unet.load_weights(
        os.path.join(PRJ_DIR, SAVED_MODEL_FILENAME) #MODIFIED FOR USER EXP KUSO
        #os.path.join(PRJ_DIR, "last_unet_video_segment.hdf5")   # SAVED_MODEL_FILENAME
    )

    # results = model_unet.evaluate(db_init, db_segmented, batch_size=BATCH_SIZE)
    # print("test loss, test acc:", results)

    # Test on new images
    # target size -> TARGET_RESOLUTION_IMAGE
    # test_path = r"db/WOW_25_exp/visual_image/1.pgm"
    # test_path = r"/home/nicolasu/Documents/Diploma Picture sets/2000_hugo_25/2000_NORMAL/2000.pgm"
    # test_path = r"db/10000/HUGO-25/10000_HUGO-25/9501.pgm"

    # test_path = r"db/10000/10000_NORMAL/9501.pgm"
    test_path = r"db/WOW_10_exp/visual_image/1.pgm"

    test_image = imread(test_path)
    test_image_rescaler = resize(test_image, TARGET_RESOLUTION_IMAGE, anti_aliasing=True)  # preserve_range=True
    test_image_rescaler = np.expand_dims(test_image_rescaler, axis=0)
    test_prediction = model_unet.predict(test_image_rescaler)

    # Threshold for predicted image
    threshold = 0.25
    test_prediction_thresholded = np.where(test_prediction > threshold, 1, 0)

    plt.figure(figsize=(15, 5))

    #plt.figure(1)
    plt.subplot(131)
    plt.imshow(test_image_rescaler.squeeze())
    plt.title('Input image')

    plt.subplot(132)
    plt.imshow(test_prediction.squeeze())
    plt.title('Predicted positions')
    plt.colorbar(shrink=0.5)

    plt.subplot(133)
    plt.imshow(test_prediction_thresholded.squeeze(), cmap='gray')
    plt.title(f'Thresholded Prediction (Threshold = {threshold})')

    plt.tight_layout()
    # plt.show()
    print("Prediction image shown.")

    # visualize results (visually compare initial and predicted segmentations)
    idx_test = 0
    predicted_segment = model_unet.predict(np.array(np.load(filename_data[idx_test])['arr_0'])[np.newaxis, ...])
    # predicted_segment = model_unet.predict(db_init[idx_test, ...][np.newaxis, ...])

    # Apply the threshold ###################################################################
    # threshold = 0.6  # Set this to your desired threshold
    # predicted_segment = (predicted_segment > threshold).astype('int')
    #########################################################################################

    ### WARNING UNTESTED SECTION ###
    '''
    # initialize counters
    total_intersection = 0
    total_union = 0

    # loop over all batches in the test set
    for idx_test in range(len(filename_data)):
        # get the model's predictions
        predicted_segment = model_unet.predict(np.array(np.load(filename_data[idx_test])['arr_0'])[np.newaxis, ...])

        # apply the threshold
        threshold = 0.6
        thresholded_predictions = (predicted_segment > threshold).astype(np.int)

        # load the true segmentations
        true_segmentations = np.array(np.load(filename_label[idx_test])['arr_0'])

        # calculate the IoU for this batch and add to the counters
        intersection = np.logical_and(true_segmentations, thresholded_predictions)
        union = np.logical_or(true_segmentations, thresholded_predictions)
        total_intersection += np.sum(intersection)
        total_union += np.sum(union)

    # calculate the average IoU across the entire test set
    average_iou_score = total_intersection / total_union

    # print the result
    print("Average IoU score with threshold {}: {}".format(threshold, average_iou_score))
    '''
    ### END OF UNTESTED SECTION ###


    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model_unet.evaluate(generator_evaluate, batch_size=BATCH_SIZE)
    print("test loss, test acc:", results)

    print("END of EVAL")
    print("EXIT")


if __name__ == "__main__":

    # sources
    # https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
    # https://github.com/qubvel/segmentation_models

    # prepare db (on demand)
    if False:
        prepare_db()
    else:
        filename_data = [os.path.join(DB_DATA, curr_filename) for curr_filename in os.listdir(DB_DATA)]
        filename_label = [os.path.join(DB_LABEL, curr_filename) for curr_filename in os.listdir(DB_LABEL)]

    # train_wrapper(filename_data, filename_label)
    evaluate_wrapper(filename_data, filename_label)

    print("All is OK")
