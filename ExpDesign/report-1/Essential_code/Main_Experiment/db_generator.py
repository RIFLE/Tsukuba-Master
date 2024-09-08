import numpy as np
import tensorflow as tf


class DBGenerator(tf.keras.utils.Sequence):

    def __init__(self, data_filenames, labels, batch_size):
        self.data_filenames = data_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.data_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.data_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        data = np.array([np.load(file_name)['arr_0'] for file_name in batch_x])
        label = np.array([np.load(file_name)['arr_0'] for file_name in batch_y])

        return data, label
