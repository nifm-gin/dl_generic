import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, Concatenate, Conv3DTranspose, Conv2D, MaxPool2D, Conv2DTranspose, ReLU, LeakyReLU, Add, Reshape, Flatten, Dense
import os
import numpy as np
from tensorflow.keras import Model, Input
import sys
import argparse
from Models import Model3D
from ops import vae_sigmoid_loss_logits
####################################################################################################

# UNET model #

####################################################################################################

class VAE(Model3D):
    """
    Unet model that heritates from Models class  
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 3, dropout=.1, batch_norm=False,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "vae", padding='same', strides = 1, weights_path = None,
                 save_path = None, activation = "relu", depth = 4, features_factor = 1, attention = False, name = "VAE", latent_dim = 2, contrast_encoder = False):
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
        self.latent_dim = latent_dim
        self.features_factor = features_factor
        self.depth = depth
        self.loss = vae_sigmoid_loss_logits
        self.contrast_encoder = contrast_encoder
        super().__init__(input_shape, data_path, num_channels, model, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation, save_path = save_path, name = name)
        self.model.compile(loss = "mae",
                           optimizer = self.optimizer, metrics = ["accuracy"])
        #self.model.summary()
            
    def build_model(self):
        """
        Method building our VAE model,
        is adapted to 2d and 3d architecture
        """
        #init
        inputs = Input(self.input_shape + (self.num_channels,))
        contrast_inputs = Input(self.input_shape + (self.num_channels,))
        x_contrast = None
        #Build encoder
        print(self.contrast_encoder)
        ff = self.features_factor
        x = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init, activation = "relu")(inputs)
        for d in range(1, self.depth):
            x = self.conv(ff * 2 ** d, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init, activation = "relu")(x)
        x = self.conv(ff * 2 ** (self.depth - 1), self.filter_shape, strides = 1, padding = self.padding, kernel_initializer = self.weights_init, activation = "relu")(x)
        if self.contrast_encoder :
            x_contrast = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init, activation = "relu")(contrast_inputs)
            for d in range(1, self.depth):
                x_contrast = self.conv(ff * 2 ** d, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init, activation = "relu")(x_contrast)
            x_contrast = self.conv(ff * 2 ** (self.depth - 1), self.filter_shape, strides = 1, padding = self.padding, kernel_initializer = self.weights_init, activation = "relu")(x_contrast)
            x_contrast = Dense(ff * 2 ** (self.depth - 1),kernel_initializer = self.weights_init, activation = "relu")(x_contrast)
            x_contrast = Dense(ff * 2 ** (self.depth - 1), kernel_initializer = self.weights_init, activation = "relu")(x_contrast)
            x_contrast = Dense(ff * 2 ** (self.depth - 1), kernel_initializer = self.weights_init, activation = "relu")(x_contrast)
        # last_conv_shape = tf.shape(x)
        #Flatten if we want a dense VAE
        #x = Flatten()(x)
        #x = Dense(self.latent_dim + self.latent_dim)(x)

        mean, logvar = tf.split(x, num_or_size_splits=2, axis=-1)
        epsilon = tf.random.normal(shape=mean.shape[1:])
        anat_reparameterized = epsilon * tf.exp(logvar * .5) + mean

        mean_c, logvar_c = tf.split(x_contrast, num_or_size_splits=2, axis=-1)
        epsilon_c = tf.random.normal(shape=mean_c.shape[1:])
        cont_reparameterized = epsilon_c * tf.exp(logvar_c * .5) + mean_c
        #Build decoder
        #decoder_input_shape = [d // 2**self.depth for d in self.input_shape]
        #x = Dense(units= np.prod(decoder_input_shape) * (ff * 2 ** (self.depth - 2)))(reparameterized)
        #x = Reshape(target_shape = tuple(decoder_input_shape) + (ff * 2 ** (self.depth - 2),))(x)
        if self.contrast_encoder :
            reparameterized = Concatenate()([anat_reparameterized, cont_reparameterized])
        else:
            reparameterized =  anat_reparemeterized
        x = self.convTranspose(ff * 2**(self.depth - 1), self.filter_shape, padding = self.padding, strides = 1, kernel_initializer = self.weights_init, activation = 'relu')(reparameterized)
        for d in range(self.depth - 2, -2, -1):
            x = self.convTranspose(ff * 2**d, self.filter_shape, padding = self.padding, strides = self.strides, kernel_initializer = self.weights_init, activation = 'relu')(x)

        x = self.convTranspose(1, self.filter_shape, padding = self.padding, strides = 1, kernel_initializer = self.weights_init)(x)
        #Update self instance with built model
        if self.contrast_encoder :
            self.model = Model(inputs = [inputs, contrast_inputs], outputs = [x, reparameterized, mean, logvar, anat_reparameterized], name = self.name)
        else:
            self.model = Model(inputs = inputs, outputs = [x, reparameterized, mean, logvar, anat_reparameterized], name = self.name)
        self.features_extractor = Model(inputs = [inputs, contrast_inputs] if self.contrast_encoder else inputs , outputs = Flatten()(reparameterized), name = self.name + "_extractor")
        self.model.summary()
        
    @tf.function
    def train_step(self, inp, y, epoch, write = False):
        #Shifting input and target doesnt make sense here as our images always have the same dimensions
        with tf.GradientTape() as tape:
            predictions, z, mean, logvar = self.model(inp, training = True)
            loss = self.loss(predictions, inp, z,mean, logvar)
        self.train_loss(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if write:
            with self.writer.as_default():
                tf.summary.scalar('training_loss', self.train_loss.result(), step=epoch)
                try :
                    tf.summary.scalar("LR", self.optimizer.lr.lr, step = epoch)
                except:
                    tf.summary.scalar("LR", self.optimizer.lr, step = epoch)
                for t in gradients :
                    tf.summary.histogram("Layer_%s " % t.name, data=t, step = epoch)

    @tf.function
    def val_step(self, inp, tar, epoch, on_cpu):
        with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
            predictions, z, mean, logvar = self.model(inp, training = False)
            
            loss = self.loss(predictions, inp, z, mean, logvar)
            with self.writer.as_default():
                tf.summary.scalar('Val_loss', loss, step=epoch)
            return loss

    def data_generator(self, data_type = 'train'):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        while True:
            indexes = np.random.choice(num_patches, self.batch_size)
            X = np.empty((self.batch_size, *self.input_shape, self.num_channels))            
            # Generate data
            for i, ID in enumerate(indexes):
                 #load patches
                X[i] = np.load("%s/%s/%s.npy" % (self.data_path, data_type, ID)).reshape((*self.input_shape, self.num_channels))
            yield X, None
            
    def generator_val(self):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/val/nb_patches.npy" % (self.data_path))
        X = np.empty((num_patches, *self.input_shape, self.num_channels))
        for i in range(num_patches):
            X[i] = np.load("%s/val/%s.npy" % (self.data_path, i)).reshape((*self.input_shape, self.num_channels))          
        return X, None

    def infer(self, is_gan = False, saliency = False):
        """
        method infering test data with our VAE model
        Infering data present in test directory, subject wisely
        """
        for subject_id in os.listdir(self.data_path + "test/"):
            print("Inferring subject %s" % subject_id)
            num_patches = np.load("%s/test/%s/nb_patches.npy" % (self.data_path, subject_id))
            X = np.empty((num_patches, *self.input_shape, self.num_channels))
            y = np.empty((num_patches, *self.input_shape, 1))
            
            # Generate data
            for i in range(num_patches):
                #load patches
                X[i] = np.load("%s/test/%s/%s.npy" % (self.data_path, subject_id, i)).reshape((*self.input_shape, self.num_channels))
            y, _, _, _ = self.model.predict(X)
            y = tf.sigmoid(y)
            if not os.path.exists("%s/infered/%s/" % (self.save_path, subject_id)):
                os.makedirs("%s/infered/%s/" % (self.save_path, subject_id))
            for i, pred in enumerate(y):
                np.save("%s/infered/%s/output_%s.npy" % (self.save_path, subject_id, i), pred)
