# Imports

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
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
import pandas as pd
import numpy as np
import cv2
import boto3
import io


# Model Definition
def get_model(input_shape, reg, metrics):
    inp1 = Input(shape=input_shape)
    inp2 = Input(shape=input_shape)
    inp3 = Input(shape=input_shape)
    inp = Concatenate(axis=3)([inp1, inp2, inp3])

    lrelu_alpha = 0.2

    x = Conv2D(16, (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha), 
               kernel_regularizer=reg, bias_regularizer=reg)(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(32, (5, 5), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha),
              kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(64, (5, 5), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha),
              kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(128, (5, 5), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha),
              kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(256, (5, 5), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha),
              kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01))(x)

    model = Model([inp1, inp2, inp3], out)
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
    def on_epoch_end(self, batch, logs=None):
        print(f'lr: {K.get_value(self.model.optimizer.lr):.8f}')
        

# Special generator to generate the 3 parts of the input image as 3 separate input images -- from DataFrame
def df_generator(gen, dataset, dataframe_dir, target_size, batch_size, class_mode, subset):
    train_df_name = dataframe_dir.split('/')[-1] + '.csv'
    train_df_path = os.path.join(dataframe_dir, train_df_name)
    train_df = pd.read_csv(train_df_path)
    
    im_gen = gen.flow_from_dataframe(dataframe=train_df,
                                     directory=dataset,
                                     x_col='out_name',
                                     y_col='class_name',
                                     weight_col='sample_weight',
                                     target_size=target_size, 
                                     batch_size=batch_size,
                                     class_mode=class_mode,
                                     color_mode='grayscale',
                                     subset=subset)
            
    while True:
        im1_s, im2_s, im3_s = [], [], []
        images, labels, sample_weights = im_gen.next()

        for im in images:
            w = im.shape[1]
            im1 = im[:, :w//3]
            im2 = im[:, w//3:(w*2)//3] 
            im3 = im[:, (w*2)//3:] 
            im1_s.append(im1)
            im2_s.append(im2)
            im3_s.append(im3)
                            
        im1_s = np.array(im1_s)
        im2_s = np.array(im2_s)
        im3_s = np.array(im3_s)
        yield [im1_s, im2_s, im3_s], labels, sample_weights
        

# Special generator to generate the 3 parts of the input image as 3 separate input images -- from dorectory
def dir_generator(gen, dataset, target_size, batch_size, class_mode, subset):

    im_gen = gen.flow_from_directory(dataset, 
                                     target_size=target_size, 
                                     batch_size=batch_size,
                                     class_mode=class_mode,
                                     color_mode='grayscale',
                                     subset=subset)
    
    while True:
        im1_s, im2_s, im3_s = [], [], []
        images, labels = im_gen.next()

        for im in images:
            w = im.shape[1]
            im1 = im[:, :w//3]
            im2 = im[:, w//3:(w*2)//3] 
            im3 = im[:, (w*2)//3:] 
            im1_s.append(im1)
            im2_s.append(im2)
            im3_s.append(im3)
                            
        im1_s = np.array(im1_s)
        im2_s = np.array(im2_s)
        im3_s = np.array(im3_s)
        yield [im1_s, im2_s, im3_s], labels
        

def main(args):
    global base_lr
    
    # Hyper-parameters
    epochs       = args.epochs
    base_lr      = args.learning_rate
    batch_size   = args.batch_size
    optimizer    = args.optimizer
    model_dir    = args.model_dir

    # SageMaker input channels
    training_dataset = args.training
    eval_dataset = args.eval
    dataframe_dir = args.dataframe_dir
    
    # Constant training params
    epochs = args.epochs
    batch_size = args.batch_size
    reg = l1_l2(0.01)

    # Metrics Defiition
    metrics = [
          tf.keras.metrics.AUC(name='auc'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.SpecificityAtSensitivity(0.9, name='specifity')
    ]    
 
    # Callback Definitions
    
    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        verbose=1,
        patience=2,
        mode='min',
        restore_best_weights=True)
    
    # Learning Rate
    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    
    # Loss History
    loss_history = LossHistory()
    
    # Checkpoint
    checkpoint_prefix = 's3://obstacles-classification-model-checkpoints'
    model_name = model_dir.split('/')[-3]
    model_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
    
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            checkpoint_filepath = f'{checkpoint_prefix}/{model_name}/{model_time}/epoch-{epoch+1}'
            model.save(checkpoint_filepath, save_format='tf')
            
    checkpoint = MyCallback()
    
    # List of all defined callbacks
    callbacks = [early_stopping, lrate, loss_history, checkpoint]

    ### End of Callback Definitions

    img_height, img_width = 200, 600
    input_shape = (img_height, img_width//3, 1)
       
    # Model definition
    model = get_model(input_shape, reg, metrics)
    
    # Data Generators drefinition
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1. / 255
    )

    training_generator = df_generator(
        datagen,
        training_dataset,
        dataframe_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    validation_generator = df_generator(
        datagen,
        training_dataset,
        dataframe_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')

    eval_generator = dir_generator(
        datagen,
        eval_dataset,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',   
        subset=None)

    # Calculate the number of steps per epoch
    train_df_name = dataframe_dir.split('/')[-1] + '.csv'
    train_df_path = os.path.join(dataframe_dir, train_df_name)
    train_df = pd.read_csv(train_df_path)
    all_images = train_df['out_name'].shape[0]
    train_images = int(all_images * 0.8)
    validation_images = all_images - train_images
    train_mod = 0 if train_images % 32 == 0 else 1
    validation_mod = 0 if validation_images % 32 == 0 else 1
    steps_per_epoch = train_images//batch_size +  train_mod
    validation_steps = validation_images//batch_size + validation_mod
    
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
    parser.add_argument('--dataframe_dir', type=str)

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--training',         type=str,   default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()
    main(args)
