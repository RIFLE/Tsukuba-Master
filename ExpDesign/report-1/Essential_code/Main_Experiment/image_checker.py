
import os

import numpy as np

from matplotlib import pyplot as plt

from settings import TARGET_RESOLUTION_IMAGE, PRJ_DIR

from skimage.io import imread
from skimage.transform import resize

from utils import init_model

model_unet, model_preprocess_input = init_model()
is_color = True


def compute_iou(prediction, ground_truth):
    # Normalizing the ground truth image to also have values 0 and 1.
    ground_truth = ground_truth / 255

    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def compute_accuracy(prediction, ground_truth):
    # Normalizing the ground truth image to also have values 0 and 1.
    ground_truth = ground_truth / 255

    total = np.size(ground_truth)
    correct = np.sum(prediction == ground_truth)

    accuracy = correct / total
    return accuracy

def pred_result(test_path, ground_truth_path, model_filename, threshold=0.25, figure_name="default"):

    #BACKUP#BACKUP#BACKUP#BACKUP#BACKUP#BACKUP#BACKUP#

    # backup_file = f"{os.path.join(PRJ_DIR, SAVED_MODEL_FILENAME)}_{datetime.now().strftime('%Y%m%d%H%M%S')}.backup.hdf5"
    # shutil.copy2(SAVED_MODEL_FILENAME,backup_file)
    # print(f"Backup of {SAVED_MODEL_FILENAME} saved as {backup_file}")

    # prepare model
    model_unet.load_weights(
        os.path.join(PRJ_DIR, model_filename) #MODIFIED FOR USER EXP KUSO
        # os.path.join(PRJ_DIR, "last_unet_video_segment.hdf5")   # SAVED_MODEL_FILENAME
    )

    # results = model_unet.evaluate(db_init, db_segmented, batch_size=BATCH_SIZE)
    # print("test loss, test acc:", results)

    # Test on new images
    # target size -> TARGET_RESOLUTION_IMAGE
    # test_path = r"db/WOW_25_exp/visual_image/1.pgm"
    # test_path = r"/home/nicolasu/Documents/Diploma Picture sets/2000_hugo_25/2000_NORMAL/2000.pgm"
    # test_path = r"db/10000/HUGO-25/10000_HUGO-25/9501.pgm"

    # test_path = r"db/10000/10000_NORMAL/9501.pgm"

    # test_path = r"db/10000/HUGO-25/10000_HUGO-25/1.pgm"

    test_image = imread(test_path)
    test_image_rescaler = resize(test_image, TARGET_RESOLUTION_IMAGE, anti_aliasing=True)  # preserve_range=True
    test_image_rescaler = np.expand_dims(test_image_rescaler, axis=0)
    test_prediction = model_unet.predict(test_image_rescaler)

    # Threshold for predicted image

    test_prediction_thresholded = np.where(test_prediction > threshold, 1, 0)

    # Read the ground truth image
    ground_truth_image = imread(ground_truth_path)

    iou_score = compute_iou(test_prediction_thresholded.squeeze(), ground_truth_image)
    print("IoU score:", iou_score)

    accuracy_score = compute_accuracy(test_prediction_thresholded.squeeze(), ground_truth_image)
    print("Accuracy score:", accuracy_score)

    plt.figure(figsize=(20, 5), num=figure_name)

    #plt.figure(1)
    plt.subplot(141)
    plt.imshow(test_image_rescaler.squeeze())
    plt.title('Input image')

    plt.subplot(142)
    plt.imshow(test_prediction.squeeze())
    plt.title('Predicted positions')
    plt.colorbar(shrink=0.625)

    plt.subplot(143)
    plt.imshow(test_prediction_thresholded.squeeze(), cmap='gray')
    plt.title(f'Thresholded Prediction (Threshold = {threshold})')

    # Adding accuracy and IoU to the plot
    # plt.text(0.15, 0.91, f'Labeling Accuracy:  {accuracy_score * 100:.2f}%\nIntersection/Union:  {iou_score * 100:.2f}%',
    #          bbox=dict(facecolor='white', alpha=0.5), transform=plt.gcf().transFigure)

    plt.text(0.05, 0.91, f'Labeling Accuracy:  {accuracy_score * 100:.2f}%\nIntersection/Union:  {iou_score * 100:.2f}%',
                       bbox=dict(facecolor='white', alpha=0.5), transform=plt.gcf().transFigure)

    plt.subplot(144)
    plt.imshow(ground_truth_image, cmap='gray')
    plt.title('Ground Truth')  # Title for the fourth image

    plt.suptitle(figure_name, fontsize=20)  # Add figure's name as the title of the figure
    plt.tight_layout()

    # Removing "," and replacing spaces with "-"
    clean_figure_name = figure_name.replace(",", "").replace(" ", "-")
    plt.savefig(f"figures/{clean_figure_name}.png")  # Save the figure as png
    print(f"Prediction image saved as {clean_figure_name}")

    plt.show()
    print("Prediction image shown.")

    # visualize results (visually compare initial and predicted segmentations)
    #    idx_test = 0
    #    predicted_segment = model_unet.predict(np.array(np.load(filename_data[idx_test])['arr_0'])[np.newaxis, ...])
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

    print("END of PRED")
    print("EXIT")

'''
pred_result("db/10000/WOW-25/10000_WOW-25/10000.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_04.hdf5",
            0.1)
            
pred_result("db/10000/HUGO-10/10000_HUGO-10/10000.pgm",
            "HUGO-10_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1)



pred_result("../10000/WOW-10/10000_WOW-10/10000.pgm",
            "../10000/WOW-10/10000_MASKS-WOW-10/10000.pgm",
            "WOW-10_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1,
            "WOW 10%, 9500 TRAIN EXAMPLES, 1st EPOCH, TEST IMAGE")

pred_result("../10000/WOW-10/10000_WOW-10/10000.pgm",
            "../10000/WOW-10/10000_MASKS-WOW-10/10000.pgm",
            "WOW-10_9500-500-0.85_1B_unet_video_segment_02.hdf5",
            0.1,
            "WOW 10%, 9500 TRAIN EXAMPLES, 2nd EPOCH, TEST IMAGE")

pred_result("../10000/WOW-10/10000_WOW-10/10000.pgm",
            "../10000/WOW-10/10000_MASKS-WOW-10/10000.pgm",
            "WOW-10_9500-500-0.85_1B_unet_video_segment_03.hdf5",
            0.1,
            "WOW 10%, 9500 TRAIN EXAMPLES, 3rd EPOCH, TEST IMAGE")

pred_result("../10000/WOW-10/10000_WOW-10/10000.pgm",
            "../10000/WOW-10/10000_MASKS-WOW-10/10000.pgm",
            "WOW-10_9500-500-0.85_1B_unet_video_segment_04.hdf5",
            0.1,
            "WOW 10%, 9500 TRAIN EXAMPLES, 4th EPOCH, TEST IMAGE")


pred_result("../10000/WOW-10/10000_WOW-10/1.pgm",
            "../10000/WOW-10/10000_MASKS-WOW-10/1.pgm",
            "WOW-10_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1,
            "WOW 10%, 9500 TRAIN EXAMPLES, 1st EPOCH, TRAIN IMAGE")

pred_result("../10000/WOW-10/10000_WOW-10/1.pgm",
            "../10000/WOW-10/10000_MASKS-WOW-10/1.pgm",
            "WOW-10_9500-500-0.85_1B_unet_video_segment_02.hdf5",
            0.1,
            "WOW 10%, 9500 TRAIN EXAMPLES, 2nd EPOCH, TRAIN IMAGE")

pred_result("../10000/WOW-10/10000_WOW-10/1.pgm",
            "../10000/WOW-10/10000_MASKS-WOW-10/1.pgm",
            "WOW-10_9500-500-0.85_1B_unet_video_segment_03.hdf5",
            0.1,
            "WOW 10%, 9500 TRAIN EXAMPLES, 3rd EPOCH, TRAIN IMAGE")

pred_result("../10000/WOW-10/10000_WOW-10/1.pgm",
            "../10000/WOW-10/10000_MASKS-WOW-10/1.pgm",
            "WOW-10_9500-500-0.85_1B_unet_video_segment_04.hdf5",
            0.1,
            "WOW 10%, 9500 TRAIN EXAMPLES, 4th EPOCH, TRAIN IMAGE")
            
            

pred_result("../10000/WOW-25/10000_WOW-25/10000.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/10000.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 1st EPOCH, TEST IMAGE")

pred_result("../10000/WOW-25/10000_WOW-25/10000.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/10000.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_02.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 2nd EPOCH, TEST IMAGE")

pred_result("../10000/WOW-25/10000_WOW-25/10000.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/10000.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_03.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 3rd EPOCH, TEST IMAGE")

pred_result("../10000/WOW-25/10000_WOW-25/10000.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/10000.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_04.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 4th EPOCH, TEST IMAGE")

pred_result("../10000/WOW-25/10000_WOW-25/10000.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/10000.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_05.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 5th EPOCH, TEST IMAGE")


pred_result("../10000/WOW-25/10000_WOW-25/1.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/1.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 1st EPOCH, TRAIN IMAGE")

pred_result("../10000/WOW-25/10000_WOW-25/1.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/1.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_02.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 2nd EPOCH, TRAIN IMAGE")

pred_result("../10000/WOW-25/10000_WOW-25/1.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/1.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_03.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 3rd EPOCH, TRAIN IMAGE")

pred_result("../10000/WOW-25/10000_WOW-25/1.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/1.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_04.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 4th EPOCH, TRAIN IMAGE")

pred_result("../10000/WOW-25/10000_WOW-25/1.pgm",
            "../10000/WOW-25/10000_MASKS-WOW-25/1.pgm",
            "WOW-25_9500-500-0.85_1B_unet_video_segment_05.hdf5",
            0.1,
            "WOW 25%, 9500 TRAIN EXAMPLES, 5th EPOCH, TRAIN IMAGE")
            

pred_result("../10000/HUGO-10/10000_HUGO-10/10000.pgm",
            "../10000/HUGO-10/10000_MASKS-HUGO-10/10000.pgm",
            "HUGO-10_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1,
            "HUGO 10%, 9500 TRAIN EXAMPLES, 1st EPOCH, TEST IMAGE")

pred_result("../10000/HUGO-10/10000_HUGO-10/10000.pgm",
            "../10000/HUGO-10/10000_MASKS-HUGO-10/10000.pgm",
            "HUGO-10_9500-500-0.85_1B_unet_video_segment_02.hdf5",
            0.1,
            "HUGO 10%, 9500 TRAIN EXAMPLES, 2nd EPOCH, TEST IMAGE")

pred_result("../10000/HUGO-10/10000_HUGO-10/10000.pgm",
            "../10000/HUGO-10/10000_MASKS-HUGO-10/10000.pgm",
            "HUGO-10_9500-500-0.85_1B_unet_video_segment_03.hdf5",
            0.1,
            "HUGO 10%, 9500 TRAIN EXAMPLES, 3rd EPOCH, TEST IMAGE")
            

pred_result("../10000/HUGO-10/10000_HUGO-10/1.pgm",
            "../10000/HUGO-10/10000_MASKS-HUGO-10/1.pgm",
            "HUGO-10_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1,
            "HUGO 10%, 9500 TRAIN EXAMPLES, 1st EPOCH, TRAIN IMAGE")

pred_result("../10000/HUGO-10/10000_HUGO-10/1.pgm",
            "../10000/HUGO-10/10000_MASKS-HUGO-10/1.pgm",
            "HUGO-10_9500-500-0.85_1B_unet_video_segment_02.hdf5",
            0.1,
            "HUGO 10%, 9500 TRAIN EXAMPLES, 2nd EPOCH, TRAIN IMAGE")

pred_result("../10000/HUGO-10/10000_HUGO-10/1.pgm",
            "../10000/HUGO-10/10000_MASKS-HUGO-10/1.pgm",
            "HUGO-10_9500-500-0.85_1B_unet_video_segment_03.hdf5",
            0.1,
            "HUGO 10%, 9500 TRAIN EXAMPLES, 3rd EPOCH, TRAIN IMAGE")

'''

'''




pred_result("../10000/HUGO-25/10000_HUGO-25/10000.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/10000.pgm",
            "HUGO-25_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1,
            "HUGO 25%, 9500 TRAIN EXAMPLES, BATCH=5, 1st EPOCH, TEST IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/10000.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/10000.pgm",
            "HUGO-25_9500-500-0.85_1B_unet_video_segment_02.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 2nd EPOCH, TEST IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/10000.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/10000.pgm",
            "HUGO-25_9500-500-0.85_1B_unet_video_segment_03.hdf5",
            0.1,
            "HUGO 25%, 9500 TRAIN EXAMPLES, BATCH=5, 3rd EPOCH, TEST IMAGE")



pred_result("../10000/HUGO-25/10000_HUGO-25/1.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/1.pgm",
            "HUGO-25_9500-500-0.85_1B_unet_video_segment_01.hdf5",
            0.1,
            "HUGO 25%, 9500 TRAIN EXAMPLES, BATCH=5, 1st EPOCH, TRAIN IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/1.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/1.pgm",
            "HUGO-25_9500-500-0.85_1B_unet_video_segment_02.hdf5",
            0.1,
            "HUGO 25%, 9500 TRAIN EXAMPLES, BATCH=5, 2nd EPOCH, TRAIN IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/1.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/1.pgm",
            "HUGO-25_9500-500-0.85_1B_unet_video_segment_03.hdf5",
            0.1,
            "HUGO 25%, 9500 TRAIN EXAMPLES, BATCH=5, 3rd EPOCH, TRAIN IMAGE")
'''


pred_result("../10000/HUGO-25/10000_HUGO-25/10000.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/10000.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_01.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 1st EPOCH, TEST IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/10000.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/10000.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_02.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 2nd EPOCH, TEST IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/10000.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/10000.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_03.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 3rd EPOCH, TEST IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/10000.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/10000.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_04.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 4th EPOCH, TEST IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/10000.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/10000.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_05.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 5th EPOCH, TEST IMAGE")



pred_result("../10000/HUGO-25/10000_HUGO-25/1.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/1.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_01.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 1st EPOCH, TRAIN IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/1.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/1.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_02.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 2nd EPOCH, TRAIN IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/1.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/1.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_03.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 3rd EPOCH, TRAIN IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/1.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/1.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_04.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 4th EPOCH, TRAIN IMAGE")

pred_result("../10000/HUGO-25/10000_HUGO-25/1.pgm",
            "../10000/HUGO-25/10000_MASKS-HUGO-25/1.pgm",
            "HUGO-25_9500-500-0.85_unet_video_segment_05.hdf5",
            0.1,
            "HUGO 25%, 4750 + 4750 EMPTY TRAIN EXAMPLES, BATCH=5, 5th EPOCH, TRAIN IMAGE")