import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow import keras
from keras.optimizers import SGD

from settings import TARGET_RESOLUTION_MODEL


def list2tensor(data, is_color: bool = False):
    if is_color:
        tensor = np.array(data)
    else:
        tensor = np.dstack(data)
        tensor = tensor[np.newaxis, ...]
        tensor = np.moveaxis(tensor, [0, 3], [3, 0])
    return tensor


def combined_loss_CE_DL(y_true, y_pred):
    # cross entropy and dice loss
    def dice_loss(y_true, y_pred):
      y_pred = tf.math.sigmoid(y_pred)
      numerator = 2 * tf.reduce_sum(y_true * y_pred)
      denominator = tf.reduce_sum(y_true + y_pred)

      return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)

def tversky_loss(y_true, y_pred, beta=0.5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch
    # thanks @mfernezir for catching a bug in an earlier version of this implementation!


def init_model():
    BACKBONE = 'efficientnetb3'   # resnet34 / efficientnetb3 / mobilenetv2
    print("load model with backbone {} ...".format(BACKBONE))
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # model_unet = unet(input_size=TARGET_RESOLUTION_MODEL)
    # model_unet = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid') # - the best result
    # model_unet = sm.Linknet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid') # - to be checked (trained)
    # model_unet = sm.PSPNet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')
    model_unet = sm.FPN(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')

    # loss = 'binary_crossentropy'
    # loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=False)  # gamma=2 the best result
    # loss = combined_loss_CE_DL
    # loss = tversky_loss
    # loss = soft_dice_loss
    # loss = tf.keras.losses.MeanAbsoluteError()

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # optimizer = SGD(learning_rate=1e-4)

    # metrics = ['accuracy', 'mean_absolute_error']

    loss = sm.losses.bce_jaccard_loss,
    metrics = [sm.metrics.iou_score, 'accuracy', 'mean_absolute_error']

    model_unet.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model_unet, preprocess_input