import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, Concatenate, Conv3DTranspose, Conv2D, MaxPool2D, Conv2DTranspose, ReLU, LeakyReLU, Add, Flatten
import os
import numpy as np
from tensorflow.keras import Model, Input
import sys
import argparse
from Models import Model3D
from ops import dice_loss, binary_ce
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

####################################################################################################

# UNET model #

####################################################################################################

class Unet(Model3D):
    """
    Unet model that heritates from Models class  
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 3, dropout=.1, batch_norm=False,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "segmentation", padding='same', strides = 1, weights_path = None,
                 save_path = None, activation = "relu", depth = 4, features_factor = 1, attention = False, name = "Unet", loss = "dice", data_augmentation = []):
        """
        parameters:
        ----------
        depth : int,
            depth of our model, number of conv block (and deconv block) in the encoder (decoder)
        """
        #check dimension of model :
        if len(input_shape) == 3:
            #3D model
            self.conv = Conv3D
            self.convTranspose = Conv3DTranspose
            self.maxPool = MaxPool3D
        else:
            #2D model
            self.conv = Conv2D
            self.convTranspose = Conv2DTranspose
            self.maxPool = MaxPool2D
        self.features_factor = features_factor
        self.depth = depth
        self.loss = loss
        self.attention = attention
        super().__init__(input_shape, data_path, num_channels, model, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation, save_path = save_path, name = name, data_augmentation = data_augmentation)
        if self.model_type == "segmentation" :
            if self.loss == "dice":
                self.loss = dice_loss
            else:   
                self.loss = binary_ce
        else:
            self.loss = tf.keras.losses.MAE
            
    def build_model(self):
        """
        Method building our Unet model,
        is adapted to 2d and 3d architecture
        A variant is possible when stride == 1 => we use maxpool (see conditions)
        """
        #init
        inputs = Input(self.input_shape + (self.num_channels,))
        skips = [None] * (self.depth + 2)
        #Build encoder
        ff = self.features_factor
        skips[0] = inputs
        x = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(inputs)
        if self.strides == 1:
            x = LeakyReLU()(x)
            x = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.strides == 1:
            x = self.maxPool(pool_size = self.pool_size)(x)
        skips[1] = x
        for d in range(1, self.depth + 1):
            x = self.conv(ff * 2**d, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(x)
            if self.strides == 1:
                x = LeakyReLU()(x)
                x = self.conv(ff * 2**d, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.strides == 1:
                x = self.maxPool(pool_size = self.pool_size)(x)
            skips[d + 1] = x
        bottleneck = x
        #Build decoder
        x = self.convTranspose(ff * 2**(self.depth - 1), self.filter_shape, padding = self.padding, strides = self.pool_size, kernel_initializer = self.weights_init)(bottleneck)
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Concatenate()([x, skips[self.depth]])
        if self.strides == 1:
            x = self.conv(ff * 2**(self.depth - 1), self.filter_shape, strides = self.strides, padding = self.padding, activation = self.activation, kernel_initializer = self.weights_init)(x)
            x = self.conv(ff * 2**(self.depth - 1), self.filter_shape, strides = self.strides, padding = self.padding, activation = self.activation, kernel_initializer = self.weights_init)(x)
        for d in range(self.depth - 1, -1, -1):
            if self.strides == 1:
                strides_conv_t = tuple(self.strides * self.pool_size for _ in self.input_shape)
            else:
                strides_conv_t = self.strides
            x = self.convTranspose(ff * 2**d, self.filter_shape, padding = self.padding, strides = strides_conv_t, kernel_initializer = self.weights_init)(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = ReLU()(x)
            if self.attention:
                skips[d] = self.attention_block(x, skips[d], ff * 2**d // 4)
            if d == 0 and self.model_type != "harmonization":
                break
            #if our Unet is for harmonization then we use the same architecture presented in DeepHarmony
            x = Concatenate()([x, skips[d]])  
            if self.strides == 1:
                x = self.conv(ff * 2**d, self.filter_shape, strides = self.strides, padding = self.padding, activation = self.activation, kernel_initializer = self.weights_init)(x)
                
        #Final layer
        if self.model_type == "segmentation" :
            #Sigmoid if we want to perform a segmentation
            x = self.convTranspose(1, self.filter_shape, strides = 1, padding = "same", activation = "sigmoid", kernel_initializer = self.weights_init)(x)
        elif self.model_type in  ["generation", "harmonization", "generation_bottleneck"] :
            #tanh if we want to perform a generation
            #tanh if we work with normalized data : between [-1; 1]
            x = self.conv(1, 4, strides = 1, padding = "same", activation = "tanh", kernel_initializer = self.weights_init)(x)
            #here we are looking for the residual so reduce the output interval
            #x = tf.math.scalar_mul(tf.constant(0.5, dtype = tf.float32), x)
            
        else:
            x = self.convTranspose(1, 4, strides = 2, padding = "same", activation = "linear", kernel_initializer = self.weights_init)(x)
        if self.model_type == "harmonization":
            #Here we know that background voxels dont need to be changed
            #So we force them to be equal to -1
            #Moreover they are not taken into account in loss computation
            #x = Add()([x, inputs])
            x = tf.where(tf.math.less(inputs, tf.constant(-0.995, dtype = tf.float32)), inputs, x)

    
        #Update self instance with built model
        if self.model_type == "generation_bottleneck":
            self.model = Model(inputs = inputs, outputs = [x, bottleneck], name = self.name)
        else:
            self.model = Model(inputs = inputs, outputs = x, name = self.name)
        self.features_extractor = Model(inputs = inputs, outputs = Flatten()(bottleneck), name = self.name + "_extractor")
        self.model.summary()
    
