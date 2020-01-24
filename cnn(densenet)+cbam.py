"""
This is a revised implementation from Cifar10 ResNet example in Keras:
(https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from models import resnext, resnet_v1, resnet_v2, mobilenets, inception_v3, inception_resnet_v2, densenet
from utils import lr_schedule
from my_keras_applications.densenet import DenseNet121
from my_keras_applications.unet import *
import numpy as np
import os

# Training parameters
batch_size = 128
epochs = 200
data_augmentation = False
num_classes = 7
subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy
base_model = 'resnet20'
# Choose what attention_module to use: cbam_block / se_block / None
attention_module = 'cbam_block'
model_type = base_model if attention_module==None else base_model+'_'+attention_module

depth = 20 # For ResNet, specify the depth (e.g. ResNet50: depth=50)
#model = resnet_v1.resnet_v1(input_shape=(448,448,3), depth=depth, attention_module=attention_module)
# model = resnet_v2.resnet_v2(input_shape=input_shape, depth=depth, attention_module=attention_module)   
# model = resnext.ResNext(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = mobilenets.MobileNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = inception_v3.InceptionV3(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
#model = densenet.DenseNet(input_shape=(448,448,3), classes=7, attention_module=attention_module)
#model = densenet.DenseNet(input_shape=(448,448,3), classes=7, attention_module=attention_module, nb_layers_per_block=[6,12,24,16], nb_dense_block=4)
#model = DenseNet121(input_tensor=Input(shape=(448,448,3)), include_top=True, weights=None, pooling='max', classes=7, backend=keras.backend, layers=keras.layers , models=keras.models, utils=keras.utils)

_, u_net_model = get_unet_128()
model = get_densenet_from_unet(u_net_model)

model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

csv_logger = CSVLogger('training.log')
callbacks = [checkpoint, lr_reducer, lr_scheduler, csv_logger]

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

batch_size = 8
train_generator = train_datagen.flow_from_directory(
        './train',
        target_size=(448, 448),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')



val_generator = train_datagen.flow_from_directory(
        './train',
        target_size=(448, 448),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

# Run training, with or without data augmentation.

history = model.fit_generator(train_generator,
                                    steps_per_epoch=np.ceil(float(train_generator.samples) / float(batch_size)),
                                    epochs=200,
                                    validation_data=val_generator,
                                    validation_steps=np.ceil(float(val_generator.samples) / float(batch_size)),
                                    callbacks=callbacks)


