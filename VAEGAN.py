import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, Concatenate, Conv3DTranspose, Conv2D, MaxPool2D, Conv2DTranspose, ReLU
import os
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
from tensorflow.keras import Model, Input
import argparse
import datetime
from Models import Model3D
from GAN import CGAN
from VAE import VAE
from CNN import CNN
from PatchGAN import PatchGAN
sys.path.append("/home/cackowss/dl_generic/utils/")
from ops import *
import time

####################################################################################################

# UNET model #

####################################################################################################

class VAEGAN(CGAN):
    """
    VAE-GAN model consisting in a GAN with VAE as generator
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 4, dropout=.1, batch_norm=False,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "GAN", padding='same', strides = 1,
                 weights_path_disc = None, weights_path_gen = None, weights_path = None,
                 save_path = None, activation = "relu", depth = 4, features_factor = 8, name = 'VAE-GAN', optimizer = "adam",
                 decay_rate = None, latent_dim = 2, contrast_encoder = False, data_augmentation = []):
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
        self.generator, self.discriminator = None, None
        self.depth = depth
        self.latent_dim = latent_dim
        self.contrast_encoder = contrast_encoder
        super(VAEGAN, self).__init__(input_shape, data_path, num_channels, None, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation, save_path = save_path, decay_rate = decay_rate, depth = self.depth, features_factor = features_factor, name = name, data_augmentation = data_augmentation)

        
    def build_model(self):
        #Build generator => VAE
        vae_object = VAE(self.input_shape, self.data_path, self.num_channels, self.model, self.pool_size,
                             self.filter_shape, self.dropout, self.batch_norm, self.learning_rate,
                             self.batch_size, "generator", self.padding, self.strides, None,
                             self.save_path, self.activation, depth = self.depth, features_factor = self.features_factor, latent_dim = self.latent_dim, contrast_encoder = self.contrast_encoder)
        self.generator = vae_object.model
        self.features_extractor = vae_object.features_extractor
        if self.contrast_encoder:
            gan_inputs = [Input(self.input_shape + (self.num_channels,)), Input(self.input_shape + (self.num_channels,))]
        else:
            gan_inputs = Input(self.input_shape + (self.num_channels,))
        generated_image, _, _, _, _ = self.generator(gan_inputs)
        #Building discriminator :
        self.discriminator = CNN(self.input_shape, self.data_path, self.num_channels, self.model, self.pool_size,
                                 self.filter_shape, self.dropout, self.batch_norm, 0.0002,
                                 self.batch_size, "discrimination", self.padding, self.strides, None,
                                 self.save_path, self.activation, depth = 3, features_factor = 64, num_classes = 2).model
        
        discriminator_output = self.discriminator(generated_image)
        self.model = Model(inputs = gan_inputs,
                           outputs = [generated_image, discriminator_output], name = self.model_type)

    def data_generator(self, data_type = "train"):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        model indicates if we training all the gan "g" or only the discriminator "d"
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        data_size = self.batch_size 
        X = np.empty((data_size, *self.input_shape, self.num_channels), dtype = np.float32)
        while True:
            indexes = np.random.choice(num_patches, self.batch_size)
            # Generate data
            for i, ID in enumerate(indexes):
                #load patches
                X[i] = np.load("%s/train/%s.npy" % (self.data_path, ID)).reshape((*self.input_shape, self.num_channels))
            yield X, None
            
    def generator_val(self):
        """
        iterator that feeds our model
        """
        num_patches = np.load("%s/val/nb_patches.npy" % (self.data_path))
        X = np.empty((num_patches, *self.input_shape, self.num_channels))
        for i in range(num_patches):
            X[i] = np.load("%s/val/%s.npy" % (self.data_path, i)).reshape((*self.input_shape, self.num_channels))          
        return X, None
    

    def restore_ckpt(self):
        checkpoint_path = self.save_path + "/checkpoints/train"
        ckpt = tf.train.Checkpoint(generator=self.generator,
                                   discriminator=self.discriminator,
                                   generator_optimizer=self.optimizer_gen,
                                   discriminator_optimizer=self.optimizer_disc)
        
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        
        # if a checkpoint exists, restore the latest checkpoint.
        try:
            if ckpt_manager.latest_checkpoint and load_ckpt:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                print ('Latest checkpoint restored!!')
            else:
                print("Could not find any saved checkpoints")
        except Exception as err:
            print("Could not restore latest checkpoint, continuing as if!", err)

            
    @tf.function
    def train_step(self, inp, y, epoch, write = False):
        #Shifting input and target doesnt make sense here as our images always have the same dimensions
        with tf.GradientTape(persistent = True) as gen_tape, tf.GradientTape(persistent = True) as disc_tape:
            gen_output, z, mean, logvar = self.generator(inp, training = True)
            disc_real_output = model.discriminator(input_image, training=True)
            disc_fake_output = model.discriminator(gen_output, training=True)
            generator_loss = vae_sigmoid_loss_logits(gen_output, inp, z, mean, logvar)
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
            loss = generator_loss + 100 * disc_loss
        self.train_loss(loss)
        #compute gradients
        generator_gradients = gen_tape.gradient(loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, 
                                                 model.discriminator.trainable_variables)
        #Apply gradient descent
        self.optimizer_gen.apply_gradients(zip(generator_gradients,
                                               self.generator.trainable_variables))
        self.optimizer_disc.apply_gradients(zip(discriminator_gradients,
                                                model.discriminator.trainable_variables))
        if write:
            with self.writer.as_default():
                tf.summary.scalar('training_loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('disc_loss', disc_loss, step=epoch)
                try :
                    tf.summary.scalar("LR", self.optimizer_gen.lr.lr, step = epoch)
                except:
                    tf.summary.scalar("LR", self.optimizer_gen.lr, step = epoch)
                for t in gradients :
                    tf.summary.histogram("Layer_%s " % t.name, data=t, step = epoch)

    @tf.function
    def val_step(self, inp, tar, epoch, on_cpu):
        with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
            gen_output, z, mean, logvar = self.generator(inp, training = True)
            disc_real_output = model.discriminator(input_image, training=True)
            disc_fake_output = model.discriminator(gen_output, training=True)
            generator_loss = vae_sigmoid_loss_logits(gen_output, inp, z, mean, logvar)
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
            loss = generator_loss + 100 * disc_loss
            
            with self.writer.as_default():
                tf.summary.scalar('total_val_loss', loss, step=epoch)
                tf.summary.scalar('disc_val_loss', disc_loss, step=epoch)
                
            return [loss, disc_loss]


