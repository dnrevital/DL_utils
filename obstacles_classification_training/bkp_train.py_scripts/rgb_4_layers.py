# Imports

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
import os
import argparse
import time
import math
import numpy as np
import cv2
import boto3
import io


# Model Definition
def get_model(input_shape, reg, metrics):
    inp1 = Input(shape=input_shape)
    inp2 = Input(shape=input_shape)
    inp = Concatenate(axis=3)([inp1, inp2])

    x = Conv2D(8, (3, 3), padding='same', activation = 'relu', 
               kernel_regularizer=reg, bias_regularizer=reg)(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(16, (5, 5), padding='same', activation = 'relu',
              kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(32, (5, 5), padding='same', activation = 'relu',
              kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(64, (5, 5), padding='same', activation = 'relu',
              kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01))(x)

    model = Model([inp1, inp2], out)
    optimizer = Adam(lr=base_lr)
        
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    
    model.summary()
    
    return model

def step_decay(epoch, lr=None):
    drop = 0.5
    epochs_drop = 10.0
    lrate = base_lr * math.pow(drop,  math.floor(epoch/epochs_drop))
    
    return lrate

class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        print(f'lr: {K.get_value(self.model.optimizer.lr):.8f}')
        
# Special generator to generate the 2 halves of the input image as 2 separate input images
def two_im_generator(gen, dataset, target_size, batch_size, class_mode, subset):
    
    # Some variables to support storing 1st generated images for debug purposes
    debug_images = False
    global displayed_image
    displayed_image = False
    debug_path = 's3://obstacles-classification/debug_images'
    
    im_gen = gen.flow_from_directory(dataset, 
                                     target_size=target_size, 
                                     batch_size=batch_size,
                                     class_mode=class_mode,
                                     subset=subset)
    
    while True:
        im1_s, im2_s = [], []
        images, labels = im_gen.next()

        for im in images:
            w = im.shape[1]
            im1 = im[:,:w//2]
            im2 = im[:,w//2:] 
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            im1_s.append(im1)
            im2_s.append(im2)
            
            if debug_images and not displayed_image:
                bucket='obstacles-classification'
                key= 'debug_images/im1.jpg'
                client = boto3.client('s3')
                im1_255 = im1 * 255.0
                print()
                print('************* im1 *************')
                print(im1_255[100][90:110])
                print('*******************************')
                print()
                _,encoded = cv2.imencode('.jpg', im1_255)
                im_iobuf = io.BytesIO(encoded)
                client.put_object(Body=im_iobuf, Key=key, Bucket=bucket)
                key= 'debug_images/im2.jpg'
                im2_255 = im2 * 255.0
                print()
                print('************* im2 *************')
                print(im2_255[100][90:110])
                print('*******************************')
                print()
                _,encoded = cv2.imencode('.jpg', im2_255)
                im_iobuf = io.BytesIO(encoded)
                client.put_object(Body=im_iobuf, Key=key, Bucket=bucket)
                displayed_image = True
                
        im1_s = np.array(im1_s)
        im2_s = np.array(im2_s)
        yield [im1_s, im2_s], labels

def main(args):
    global base_lr
    
    # Hyper-parameters
    epochs       = args.epochs
    base_lr      = args.learning_rate
    batch_size   = args.batch_size
    optimizer    = args.optimizer

    # SageMaker input channels
    training_dataset = args.training
    eval_dataset = args.eval
    
    # Constant training params
    epochs = args.epochs
    batch_size = args.batch_size
    reg = l1_l2(0.01)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        verbose=1,
        patience=10,
        mode='min',
        restore_best_weights=True)
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    loss_history = LossHistory()
    callbacks = [early_stopping, lrate, loss_history]

    img_height, img_width = 200, 400
    input_shape = (img_height, img_width//2, 3)
    
    metrics = [
          tf.keras.metrics.AUC(name='auc'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.SpecificityAtSensitivity(0.9, name='specifity')
    ]    
    
    # Model definition
    model = get_model(input_shape, reg, metrics)
    
    # Data Generators drefinition
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1. / 255
    )

    training_generator = two_im_generator(
        datagen,
        training_dataset,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    validation_generator = two_im_generator(
        datagen,
        training_dataset,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')

    eval_generator = two_im_generator(
        datagen,
        eval_dataset,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',   
        subset=None)

    # Calculate the number of steps per epoch
    train_it = DirectoryIterator(training_dataset,
                                 image_data_generator=datagen,
                                 subset='training')
    val_it = DirectoryIterator(training_dataset,
                               image_data_generator=datagen,
                               subset='validation')
    steps_per_epoch = train_it.__len__()
    validation_steps= val_it.__len__()
    
    model.fit(training_generator,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=validation_generator,
              validation_steps=validation_steps,
              callbacks=callbacks,
              verbose=2)
    
    # Evaluate model performance
    score = model.evaluate_generator(
        eval_generator,
        steps=300,
        verbose=1)
    
    print(f'test_auc: {score[1]}')
    print(f'test_recall: {score[2]}')
    print(f'test_specifity: {score[3]}')
    
    # Save model to model directory
    model.save(f'{os.environ["SM_MODEL_DIR"]}/{time.strftime("%m%d%H%M%S", time.gmtime())}', save_format='tf')


#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--optimizer',     type=str,   default='Adam')

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--training',         type=str,   default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()
    main(args)
