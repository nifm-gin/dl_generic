import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.initializers import TruncatedNormal, GlorotNormal, GlorotUniform
from tensorflow.keras.layers import Conv3D, Concatenate, Conv2D, Flatten, Dense, Dropout
import numpy as np
from tensorflow.keras import Model, Input
import sys
from CNN import CNN
from layers import residual_block
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

####################################################################################################

# CNN model #

####################################################################################################

class Resnet(CNN):
    """
    Resnet model that heritates from Models class  
    
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 3, dropout=0.4, batch_norm=True,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "classification", padding='same', strides = 1, weights_path = None,
                 save_path = None, activation = "leakyRelu", depth = 4, features_factor = 16, num_classes = 1, name = "Resnet", class_weights = None,
                 additional_inputs = None, data_augmentation = [], gt_path = None):

        """
        parameters:
        ----------
        depth : int,
            depth of our model, number of conv block (and deconv block) in the encoder (decoder)
        """
        super().__init__(input_shape, data_path, num_channels, model, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate,
                         batch_size, model_type, padding, strides, weights_path, activation = activation, save_path = save_path,
                         depth = depth, features_factor = features_factor, num_classes = num_classes, name = name,
                         class_weights = class_weights, additional_inputs = additional_inputs, data_augmentation = data_augmentation, gt_path = gt_path)

        self.optimizer = [tf.keras.optimizers.SGD(self.learning_rate)]
        #self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.8)
        
    def build_model(self, use_pretrained = False):
        """
        Builds our classical CNN architecture
        Is adapted to being used as the discriminator of a cGAN => condition on model_type
        """
        #for this CNN with change initialization otherwise bound to be stuck in a local minimum
        self.weights_init = TruncatedNormal(0., 1)
        #self.weights_init = GlorotUniform()
        inputs = Input(self.input_shape + (self.num_channels,))
        if use_pretrained:
            self.model = tf.keras.Sequential()
            resnet = ResNet50(include_top = False, input_shape = self.input_shape + (self.num_channels,), classes = self.num_classes)
            for layer in resnet.layers:
                layer.trainable = False
            self.model.add(resnet)
            self.model.add(Flatten())
            self.model.add(Dense(1024))
            self.model.add(Dense(1, activation='linear'))
        else:
            #for this CNN with change initialization otherwise bound to be stuck in a local minimum

            #self.weights_init = TruncatedNormal(0., 0.2)
            self.weights_init = GlorotUniform()
            
            x = residual_block(inputs, self.conv, self.features_factor, reduce = True)
            x = residual_block(x, self.conv, self.features_factor)
            for i in range(1, self.depth):
                x = residual_block(x, self.conv, self.features_factor * 2**i, reduce = True)
                x = residual_block(x, self.conv, self.features_factor * 2**i)
            features = Flatten()(x)
            #x = Dropout(self.dropout)(x)
            x = Dense(128, activation='relu')(features)
            if self.additional_inputs is not None :
                clinical_inputs = Input(len(self.additional_inputs))
                x = Concatenate()([x, clinical_inputs])
                inputs = [inputs, clinical_inputs]
            #in both cases we do not use  final activation as we  will  use from_logits = True in loss_functions
            if self.num_classes == 2:
                x = Dense(1, activation='linear', kernel_initializer = self.weights_init)(x)
            else:
                #classification with n_class > 2
                x = Dense(self.num_classes, activation='linear', kernel_initializer = self.weights_init)(x)
            self.features_extractor = Model(inputs = inputs, outputs = features, name = self.name + "_extractor")
            self.model = Model(inputs = inputs, outputs = x, name = self.name)
        self.model.summary()
