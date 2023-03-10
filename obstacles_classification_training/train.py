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
import random


# Model Definition
def get_model(input_shapes, reg, metrics, from_chp=None):
    optimizer = Adam(lr=base_lr)

    if from_chp:
        print(f'========= loading checkpoint: {from_chp} =============')
        model = tf.keras.models.load_model(from_chp)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        print(f'========= model: {model} =============')
        return model
        
    inp1 = Input(shape=input_shapes[0])
    inp2 = Input(shape=input_shapes[0])
    inp3 = Input(shape=input_shapes[1])
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
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01))(x)

    model = Model([inp1, inp2, inp3], out)  
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    model.summary()
    
    return model

def step_decay(epoch, lr=None):
    drop = 0.5
    epochs_drop = 10.0
    lrate = base_lr * math.pow(drop,  math.floor(epoch/epochs_drop))
    
    return lrate

class CustomDataGenerator(ImageDataGenerator):
    def __init__(self,
                 augments={'horizontal_flip': 0.1,
                           'rotate': 0.0,
                           'mix_channels': 0.1},
                 debug_images=False,
                 **kwargs):
        self.imcount = 0
        self.bucket = 'obstacles-classification'
        self.key_prefix = 'debug_images/test_3/'
        self.client = boto3.client('s3')
        self.augments = augments
        self.debug_images = debug_images

        r, g, b = 0, 1, 2

        self.channels = {0: [r, b, g],
                         1: [g, r, b],
                         2: [g, b, r],
                         3: [b, r, g],
                         4: [b, g, r]} 
        
        super().__init__(preprocessing_function=self.my_augments,
                         **kwargs)
        
    def my_augments(self, image):
        im = np.array(image)
        any_augment = False
        rand = random.randint(1, 100)/100.
        if rand < self.augments['horizontal_flip']:
            any_augment = True
            # Divide im to original components (ref, current, mask)
            w = im.shape[1]
            ref = im[:, :w//3]
            current = im[:, w//3:(w*2)//3] 
            mask = im[:, (w*2)//3:] 
            # Horizontal Flip
            ref = cv2.flip(ref, 1)
            current = cv2.flip(current, 1)
            mask = cv2.flip(mask, 1)
            im = cv2.hconcat([ref, current, mask])            
        rand = random.randint(1, 100)/100.
        if rand < self.augments['rotate']:
            any_augment = True
            # Divide im to original components (ref, current, mask)
            w = im.shape[1]
            ref = im[:, :w//3]
            current = im[:, w//3:(w*2)//3] 
            mask = im[:, (w*2)//3:] 
            # Horizontal Flip
            ref = cv2.rotate(ref, cv2.ROTATE_180)
            current = cv2.rotate(current, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
            im = cv2.hconcat([ref, current, mask])            
        rand = random.randint(1, 100)/100.
        if rand < self.augments['mix_channels']:
            any_augment = True
            aug = int(rand*100) % 5
            rgb = cv2.split(im)
            im = cv2.merge([rgb[self.channels[aug][0]],
                            rgb[self.channels[aug][1]],
                            rgb[self.channels[aug][2]]])
        if self.debug_images and any_augment:
            self.upload_image(im)
        return im
        
    def upload_image(self, image):
        self.imcount += 1
        imname = f'test_3_{self.imcount}.jpg'
        key = self.key_prefix + imname
        im = np.array(image)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
        _,encoded = cv2.imencode('.jpg', im)
        im_iobuf = io.BytesIO(encoded)
        self.client.put_object(Body=im_iobuf, Key=key, Bucket=self.bucket)

class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        print(f'lr: {K.get_value(self.model.optimizer.lr):.8f}')
        

# Special generator to generate the 3 parts of the input image as 3 separate input images -- from DataFrame
def df_generator(gen,
                 dataset,
                 dataframe_dir,
                 target_size,
                 comp_target_size,
                 batch_size,
                 class_mode,
                 subset,
                 black_mask_augment=1.,
                 ref_as_current_augment=1.):
    
    train_df_name = dataframe_dir.split('/')[-1] + '.csv'
    train_df_path = os.path.join(dataframe_dir, train_df_name)
    train_df = pd.read_csv(train_df_path)
    imcount = 0
    bucket = 'obstacles-classification'
    key_prefix = 'debug_images/test_3/'
    client = boto3.client('s3')
    
    im_gen = gen.flow_from_dataframe(dataframe=train_df,
                                     directory=dataset,
                                     x_col='out_name',
                                     y_col='class_name',
                                     weight_col='sample_weight',
                                     batch_size=batch_size,
                                     target_size=target_size,
                                     class_mode=class_mode,
                                     subset=subset)
            
    while True:
        ref_s, current_s, mask_s = [], [], []
        images, labels, sample_weights = im_gen.next()

        for i, im in enumerate(images):
            imarr = np.array(im, dtype='float32')
            w = imarr.shape[1]
            im1 = imarr[:, :w//3]
            im2 = imarr[:, w//3:(w*2)//3] 
            im3 = imarr[:, (w*2)//3:] 
            
            # Replace mask by a black mask according to black_mask_augment probability
            rand = random.randint(1, 100)/100.
            if rand < black_mask_augment:
                h_mask = im3.shape[0]
                w_mask = im3.shape[1]
                mask = np.full((h_mask, w_mask, 3), 0, dtype=np.float32)
            else:
                mask = np.array(im3)
                
            # If the label is 1 ("obstacle") copy current to ref according to ref_as_current_augment probability
            rand = random.randint(1, 100)/100.
            cls = int(labels[i])
            if rand < ref_as_current_augment:
                ref = np.array(im2)
            else:
                ref = np.array(im1)
                
            current = np.array(im2)          
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            ref /= 255.0
            current /= 255.0
            mask /= 255.0

            ref_s.append(ref)
            current_s.append(current)
            mask_s.append(mask)
                            
        ref_s = np.array(ref_s)
        current_s = np.array(current_s)
        mask_s = np.array(mask_s)
        yield [ref_s, current_s, mask_s], labels, sample_weights
        

# Special generator to generate the 3 parts of the input image as 3 separate input images -- from directory
def dir_generator(gen, dataset, target_size, batch_size, class_mode, subset):

    im_gen = gen.flow_from_directory(dataset, 
                                     target_size=target_size, 
                                     batch_size=batch_size,
                                     class_mode=class_mode,
                                     subset=subset)
    
    while True:
        im1_s, im2_s, im3_s = [], [], []
        images, labels = im_gen.next()

        for im in images:
            imarr = np.array(im, dtype='float32')
            w = imarr.shape[1]
            im1 = imarr[:, :w//3]
            im2 = imarr[:, w//3:(w*2)//3] 
            im3 = imarr[:, (w*2)//3:] 

            #im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            #im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            im3 = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)

            im1 /= 255.0
            im2 /= 255.0
            im3 /= 255.0

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
    from_chp     = args.from_chp
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
    x3_input_shape = (img_height, img_width//3, 3)
    x1_input_shape = (img_height, img_width//3, 1)
       
    # Load Model
    model = get_model([x3_input_shape, x1_input_shape], reg, metrics, from_chp=from_chp)
    
    # Data Generators drefinition
    train_datagen = CustomDataGenerator(validation_split=0.2)
    val_datagen = ImageDataGenerator(validation_split=0.2)

    training_generator = df_generator(
        train_datagen,
        training_dataset,
        dataframe_dir,
        target_size=(img_height, img_width),
        comp_target_size=(img_height, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    validation_generator = df_generator(
        val_datagen,
        training_dataset,
        dataframe_dir,
        target_size=(img_height, img_width),
        comp_target_size=(img_height, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')

    eval_generator = dir_generator(
        val_datagen,
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
    print(f'***** steps_per_epoch: {steps_per_epoch} ********')
    print(f'***** validation_steps: {validation_steps} ********')
    
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
    parser.add_argument('--from_chp',      type=str,   default=None)
    parser.add_argument('--dataframe_dir', type=str)

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--training',         type=str,   default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()
    main(args)
