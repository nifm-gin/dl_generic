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
from Unet import Unet
from CNN import CNN
from PatchGAN import PatchGAN
sys.path.append("/home/cackowss/dl_generic/utils/")
from ops import *
import time

####################################################################################################

# UNET model #

####################################################################################################

class CGAN(Model3D):
    """
    Unet model that heritates from Models class  
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 4, dropout=.1, batch_norm=False,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "cGAN", padding='same', strides = 1,
                 weights_path_disc = None, weights_path_gen = None, weights_path = None,
                 save_path = None, activation = "relu", depth = 4, features_factor = 8, name = 'cGAN', optimizer = "adam",
                 decay_rate = None, data_augmentation = []):
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

        super(CGAN, self).__init__(input_shape, data_path, num_channels, None, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation, save_path = save_path, decay_rate = decay_rate, name = name, data_augmentation = data_augmentation)
                #Use several optimizer for each training
        #Load weights specified, separatly for D and G
        if weights_path_disc is not None:
            try:
                assert os.path.isfile(weights_path_disc), "Weights path specified for discriminator does not exist, model kept as if"
                self.discriminator.load_weights(weights_path_disc)
                print("Discriminator's weights loaded")
            except AssertionError as err:
                print(err)
            except Exception as err:
                print("Error raised while loading discriminator weights : %s.\n\tContinuing with model as if" % err)
        if weights_path_gen is not None:
            try:
                assert os.path.isfile(weights_path_gen), "Weights path specified for generator does not exist, models kept as if"
                self.generator.load_weights(weights_path_gen)
                print("Generator's weights loaded")
            except AssertionError as err:
                print(err)
            except Exception as err:
                print("Error raised while loading generator weights : %s.\n\tContinuing with model as if" % err)
        print("Init done")
        if optimizer == "RMSprop":
            self.optimizer_gen = RMSprop(learning_rate = self.learning_rate, rho = 0.9)
            self.optimizer_disc = RMSprop(learning_rate = self.learning_rate, rho = 0.9)
        else:
            self.optimizer_gen = Adam(self.learning_rate, beta_1=0.5)
            self.optimizer_disc = Adam(self.learning_rate, beta_1=0.5)
        self.optimizer = [self.optimizer_gen, self.optimizer_disc]
        #Two optimizer objects for Gen and Disc
        self.writer = tf.summary.create_file_writer(save_path + "/logs/fit/" + self.name + datetime.datetime.now().strftime("%d-%H%M"))
        self.scheduler = Scheduler(self.optimizer,
                                   names = ["Gen", "Disc"],
                                   ratios = [1, 1],
                                   patience = 3, early_stopping = 30)
        
        self.discriminator.summary()
        self.generator.summary()
        self.model.summary()
        
    def build_model(self, return_bottleneck = False):
        #Build generator => Unet
        print(self.depth)
        gen_type = "generation" + return_bottleneck * "_bottleneck"
        self.generator = Unet(self.input_shape, self.data_path, self.num_channels, self.model, self.pool_size,
                              5, self.dropout, self.batch_norm, self.learning_rate,
                              self.batch_size, gen_type, self.padding, self.strides, None,
                              self.save_path, self.activation, self.depth, self.features_factor).model
        
        gan_inputs = Input(self.input_shape + (self.num_channels,))
        generated_image = self.generator(gan_inputs)
        #Building discriminator :
        if self.model_type == "pix2pix":
            self.discriminator = CNN(self.input_shape, self.data_path, self.num_channels + 1, self.model, self.pool_size,
                                     self.filter_shape, self.dropout, self.batch_norm, 0.0002,
                                     self.batch_size, "cGAN", self.padding, self.strides, None,
                                     self.save_path, self.activation, 3, self.features_factor, 2).model
        elif self.model_type == "harmonization":
            self.discriminator = CNN(self.input_shape, self.data_path, self.num_channels, self.model, self.pool_size,
                                     self.filter_shape, self.dropout, self.batch_norm, 0.0002,
                                     self.batch_size, "classification", self.padding, self.strides, None,
                                     self.save_path, self.activation, 3, 64, 2).model
        else:
            self.discriminator = PatchGAN(self.input_shape, self.data_path, self.num_channels + 1, self.model, self.pool_size,
                                          self.filter_shape, self.dropout, self.batch_norm, 0.0002,
                                          self.batch_size, "pix2pix", self.padding, self.strides, None,
                                          self.save_path, self.activation, 3, self.features_factor, 2).model
        
        disc_input = Concatenate()([gan_inputs, generated_image])
        discriminator_output = self.discriminator(disc_input)
        self.model = Model(inputs = gan_inputs,
                           outputs = [generated_image, discriminator_output], name = self.model_type)

    def data_generator(self, data_type = "train"):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        model indicates if we training all the gan "g" or only the discriminator "d"
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        data_size = self.batch_size if data_type == "train" else num_patches
        X = np.empty((data_size, *self.input_shape, self.num_channels), dtype = np.float32)
        y = np.empty((data_size, *self.input_shape, 1), dtype = np.float32)
        while True:
            if data_type == "train":
                indexes = np.random.choice(num_patches, self.batch_size)
                # Generate data
                for i, ID in enumerate(indexes):
                    #load patches
                    X[i] = np.load("%s/train/%s.npy" % (self.data_path, ID)).reshape((*self.input_shape, self.num_channels))
                    y[i] = np.load("%s/train/output_%s.npy" % (self.data_path, ID)).reshape((*self.input_shape, 1))
            else:
                for i in range(num_patches):
                    X[i] = np.load("%s/val/%s.npy" % (self.data_path, i)).reshape((*self.input_shape, self.num_channels))
                    y[i] = np.load("%s/val/output_%s.npy" % (self.data_path, i)).reshape((*self.input_shape, 1))
            yield X, y

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

    def train(self, epochs = 10, val_on_cpu = False):
        """
        method to train our model
        """
        super(CGAN, self).train(epochs, val_on_cpu)
        self.discriminator.save_weights(self.save_path + '/best_d.h5')
        self.generator.save_weights(self.save_path + '/best_g.h5')

    @tf.function
    def train_step(self, input_image, target, epoch, write = False):
        with tf.GradientTape(persistent = True) as gen_tape, tf.GradientTape(persistent = True) as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator(tf.concat([input_image, target], axis = -1), training=True)
            disc_fake_output = self.discriminator(tf.concat([input_image, gen_output], axis = -1), training=True)
            
            # calculate the loss
            total_gen_loss, gan_loss, l1_loss = cgan_generator_loss(disc_fake_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
        
        # Calculate the gradients for generator and discriminator
        generator_gradients = gen_tape.gradient(total_gen_loss, 
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, 
                                                     self.discriminator.trainable_variables)
    
        # Apply the gradients to the optimizer
        self.optimizer_gen.apply_gradients(zip(generator_gradients, 
                                               self.generator.trainable_variables))
        self.optimizer_disc.apply_gradients(zip(discriminator_gradients,
                                                self.discriminator.trainable_variables))
        return {'total_gen_loss' : total_gen_loss,
                'disc_loss' : disc_loss}
    
    @tf.function
    def val_step_GAN(self, input_image, target, epoch, on_cpu):
        with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
            gen_output = self.generator(input_image, training=False)
            disc_real_output = self.discriminator(tf.concat([input_image, target], axis = -1), training=False)
            disc_fake_output = self.discriminator(tf.concat([input_image, gen_output], axis = -1), training=False)
        
            # calculate the loss
            total_gen_loss, gan_loss, l1_loss = cgan_generator_loss(disc_fake_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)

            return {'total_gen_val_loss' : total_gen_loss,
                    'disc_val_loss' : disc_loss}
