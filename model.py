import argparse
import csv
import os

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Conv2D, Dropout, Flatten, Dense, Lambda
from scipy import ndimage
from sklearn.model_selection import train_test_split

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, 5, strides=2, activation='relu'))
    model.add(Conv2D(36, 5, strides=2, activation='relu'))
    model.add(Conv2D(48, 5, strides=2, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def load_samples(samples, augment=True):
    """Yields augmented samples in batches of len(samples) size"""
    images = []
    angles = []

    for row in samples:
        center_img = ndimage.imread(row[0])
        center_angle = float(row[3])
        images += [center_img]
        angles += [center_angle]

        if augment:
            # Mirror the central image
            images += [cv2.flip(center_img, 1)]
            angles += [-center_angle]

            # Use the left and right camera perspective
            correction = 0.22
            left_img = ndimage.imread(row[1])
            right_img = ndimage.imread(row[2])
            left_angle = center_angle + correction
            right_angle = center_angle - correction
            images += [left_img, right_img]
            angles += [left_angle, right_angle]

    # Shuffle samples with labels
    X_train = np.array(images)
    y_train = np.array(angles)
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

    # Return augmented data in batches of the correct size
    batch_size = len(samples)
    for offset in range(0, len(X_train), batch_size):
        lo, hi = offset, offset+batch_size
        yield X_train[lo:hi], y_train[lo:hi]

def generator(samples, batch_size, augment=True):
    """Yields batches of augmented samples"""
    num_samples = len(samples)
    while 1:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            yield from load_samples(batch_samples, augment=augment)

def load_driving_logs(data_dirs):
    samples = []
    for data_dir in data_dirs:
        def norm_path(s):
            return os.path.join(data_dir, 'IMG', os.path.basename(s))
        def norm_row(row):
            return list(map(norm_path, row[:3])) + row[3:]
        with open(os.path.join(data_dir, 'driving_log.csv'), 'r') as f:
            samples += list(map(norm_row, csv.reader(f)))
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-output', default='model.h5')
    parser.add_argument('-c', '--model-checkpoint', default='model_ckpt.h5')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-E', '--early-stopping', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-l', '--plot-loss', action='store_true')
    parser.add_argument('data_dirs', nargs='+')
    args = parser.parse_args()

    # Setup the data generators
    samples = load_driving_logs(args.data_dirs)
    train_samples, validation_samples = train_test_split(samples,
                                                         test_size=0.25)
    train_generator = generator(train_samples, args.batch_size)
    validation_generator = generator(validation_samples, args.batch_size,
                                     augment=False)

    # Account for augmentations in the training dataset
    steps_per_sample = len(list(load_samples(samples[:1])))
    steps_per_epoch = np.ceil(len(train_samples) * steps_per_sample
                              / args.batch_size)

    # Note that we didn't augment the validation dataset
    validation_steps = np.ceil(len(validation_samples) / args.batch_size)

    # Configure early stopping and best model checkpointing if requested
    callbacks = []
    if args.early_stopping:
        cb = EarlyStopping(monitor='val_loss', patience=args.early_stopping,
                           verbose=1)
        callbacks.append(cb)
    if args.model_checkpoint:
        cb = ModelCheckpoint(args.model_checkpoint, monitor='val_loss',
                             mode='min', save_best_only=True, verbose=1)
        callbacks.append(cb)

    # Build and train the model
    model = build_model()
    history = model.fit_generator(
        train_generator, steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=args.epochs, verbose=1, callbacks=callbacks
    )

    if args.model_output:
        print('Saving model to {}'.format(args.model_output))
        model.save(args.model_output)

    if args.plot_loss:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

if __name__ == '__main__':
    main()
