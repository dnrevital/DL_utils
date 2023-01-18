# Imports

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
import os
import argparse
import time


def get_model(input_shape, lr, reg, metrics):
    inp = Input(shape=input_shape)

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
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(128, (5, 5), padding='same', activation = 'relu',
              kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01))(x)

    model = Model(inp, out)
    optimizer = Adam(lr=lr)
        
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    
    model.summary()
    
    return model

def main(args):
    # Hyper-parameters
    epochs       = args.epochs
    lr           = args.learning_rate
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

    img_height, img_width = 200, 200
    input_shape = (img_height, img_width, 3)
    
    metrics = [
          keras.metrics.AUC(name='auc'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.SpecificityAtSensitivity(0.9, name='specifity'),
    ]    
    
    # Model definition
    model = get_model(input_shape, lr, reg, metrics)
    
    # Data Generators drefinition
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    training_generator = datagen.flow_from_directory(
        training_dataset,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    validation_generator = datagen.flow_from_directory(
        training_dataset,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation')

    eval_generator = datagen.flow_from_directory(
        eval_dataset,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')    

    model.fit_generator(
        training_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping],
        verbose=2)
    
    # Evaluate model performance
    score = model.evaluate_generator(
        eval_generator,
        steps=30,
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
