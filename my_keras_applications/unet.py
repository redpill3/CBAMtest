from my_keras_applications.densenet import DenseNet121
from keras.losses import binary_crossentropy
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras import optimizers
import keras


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss



def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))

def get_unet_128(input_shape=(448, 448, 3),
                 num_classes=1):
    input_size = input_shape[0]

    inputs = Input(shape=input_shape)


    dense_model = DenseNet121(input_tensor=inputs, include_top=False, weights=None, pooling='max', backend=keras.backend, layers=keras.layers , models=keras.models, utils=keras.utils) 
    dense_model.layers.pop()  #remove maxpolling layer

    # (28,28) 
    up4 = UpSampling2D((2, 2))(dense_model.layers[-1].output)
    up4 = concatenate([dense_model.get_layer('conv4_block24_concat').output, up4], axis=3)
    up4 = Conv2D(256, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)


    # (56,56) 
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([dense_model.get_layer('conv3_block12_concat').output, up3], axis=3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)


    # (112,112) 
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([dense_model.get_layer('conv2_block6_concat').output, up2], axis=3)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)


    # (224,224) 
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([dense_model.get_layer('conv1/conv').output, up1], axis=3)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)


    # (448,448)
    up0 = UpSampling2D((2, 2))(up1)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)


    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    #u-net compile
    u_net_model = Model(inputs=inputs, outputs=classify)
    u_net_model.compile(optimizer=optimizers.adam(lr=0.001), loss=bce_dice_loss, metrics=[dice_loss])   
    print('u-net model : densenet encoder ')


    return input_size, u_net_model

def get_densenet_from_unet( unet_model):
  dense_input = unet_model.input
  dense_center_output = unet_model.get_layer('relu').output

  
  x = GlobalMaxPool2D()(dense_center_output)
  x = Dense(1024, name='fully', init='uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dense(512, init='uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  densenet_output = Dense(7, activation='softmax', name='softmax')(x)
  
  '''
  x = GlobalMaxPool2D()(dense_center_output)
  x = Dense(2048)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dense(1024)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  densenet_output = Dense(7, activation='softmax', name='softmax')(x)
  '''
  densenet_model = Model(inputs=dense_input, outputs=densenet_output)
  densenet_model.compile(loss=weighted_categorical_crossentropy([4.375, 2.783, 1.301, 12.440, 1.285, 0.213, 10.075]),
                  optimizer=optimizers.adam(lr=0.003),
                  metrics=['acc'])

  return  densenet_model
