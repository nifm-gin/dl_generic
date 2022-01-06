import tensorflow as tf
from tensorflow.keras.layers import Concatenate, LeakyReLU, Flatten, Dense, Dropout, Lambda
import os
import time
from tensorflow.keras.initializers import TruncatedNormal, GlorotNormal, GlorotUniform
import numpy as np
from tensorflow.keras import Model, Input
import sys
import argparse
import datetime
from CNN import CNN
from ops import binary_ce, calc_l1_loss
from layers import TransformerDecoder, TransformerEncoder
import csv
import argparse
from patches_generation import find_indices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152


####################################################################################################
 
# CNN model #

####################################################################################################

class ClassificationTransformer(CNN):
    """
    CNN model that heritates from Models class  
    
    """
    def __init__(self, input_shape, data_path, patch_shape = 16, num_channels = 1, model = None, dropout=0.4, d_model = 512, vocabulary_size = 10000,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "classification", weights_path = None, num_heads = 8,
                 save_path = None, activation = "leakyRelu", depth = 4, features_factor = 16, name = "classiformer", num_classes = 2):
        """
        parameters:
        ----------
        depth : int,
            depth of our model, number of conv block (and deconv block) in the encoder (decoder)
        """
        self.num_heads = num_heads
        self.d_model = d_model
        self.positional_encoding = 10000
        self.dff = features_factor
        self.patch_shape = tuple(patch_shape for _ in input_shape)
        self.len_patch = np.prod(np.asarray(self.patch_shape)) * num_channels
        if d_model >= self.len_patch:
            self.d_model = self.len_patch
            self.embedding = tf.identity
        else:
            self.embedding = Dense(self.d_model)
        if len(input_shape) == 2:
            self.indices = find_indices((*self.patch_shape, 1), (*input_shape, 1), save_path = None)
        else:
            self.indices = find_indices(self.patch_shape, input_shape, save_path = None)
        assert len(self.indices) >= 1, "Something went wrong in patch-disc building, verify that input dimensions are compatible with patches size %s" % self.window_size
        self.sequence_length = len(self.indices)

        super(ClassificationTransformer, self).__init__(input_shape, data_path, num_channels, model, None, None, dropout, False, 
                                                        initial_learning_rate, batch_size, model_type, 'same', 0, weights_path,
                                                        save_path, activation, depth, features_factor, num_classes, name)
        
    def build_model(self):
        """
        Builds an image transformer architecture
        """
        input_patches = Input((self.sequence_length, self.len_patch))
        self.weights_init = TruncatedNormal(0., 0.2)
        self.encoder = TransformerEncoder(self.depth, self.d_model, self.num_heads, self.dff, self.embedding, self.sequence_length)
            
        x = self.encoder(input_patches, training = True)
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        #linear activation : logits = True
        if self.num_classes == 2:
            outputs = Dense(1, kernel_initializer = self.weights_init)(x)
        else:
            outputs = Dense(self.num_classes, kernel_initializer = self.weights_init)(x)
        self.model = Model(inputs = input_patches, outputs = outputs, name = self.name)

    def get_patches(self, X):
        patches = []
        for indice in self.indices:
            if len(self.input_shape) == 2:
                patch = X[indice[0]:indice[1], indice[2]:indice[3], :]
            else:
                patch = X[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5], :]
            patch = patch.reshape(self.len_patch)
            patches.append(patch)
        return patches
    
    def data_generator(self, data_type = 'train'):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        X = np.empty((self.batch_size,
                      self.sequence_length, self.len_patch), dtype=np.float32)
        while True:
            indexes = np.random.choice(num_patches, self.batch_size)
            if self.num_classes == 2:
                y = np.zeros((self.batch_size, 1))
            else:
                y = np.zeros((self.batch_size, self.num_classes))            
            # Generate data
            for i, ID in enumerate(indexes):
                #load patches
                data = np.load("%s/%s/%s.npz" % (self.data_path, data_type, ID))
                X[i] = self.get_patches(data["data"].reshape((*self.input_shape, self.num_channels)))
                if self.num_classes == 2:
                    y[i] = data["group"]
                else:
                    y[i, data["group"]] = 1
            yield X, y
            
    def generator_val(self):
        num_patches = np.load("%s/val/nb_patches.npy" % (self.data_path))
        X = np.empty((num_patches,
                      self.sequence_length, self.len_patch), dtype=np.float32)
        if self.num_classes == 2:
                y = np.zeros((num_patches, 1))
        else:
            y = np.zeros((num_patches, self.num_classes)) 
        for i in range(num_patches):
            data = np.load("%s/val/%s.npz" % (self.data_path, i))
            X[i] = self.get_patches(data["data"].reshape((*self.input_shape, self.num_channels)))
            if self.num_classes == 2:
                y[i] = data["group"]
            else:
                y[i, data["group"]] = 1
        return X, y
            
    def infer(self):
        """
        method infering test data with our model
        Infering data present in test directory, subject wisely
        """
        if not os.path.exists("%s/infered/" % (self.save_path)):
            os.makedirs("%s/infered/" % (self.save_path))
        csv_name = "%s/infered/predictions.csv" % self.save_path
        with open(csv_name, 'w', newline='') as csvfile:
            fieldnames = ['subject_id', 'group', "prob"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for subject_id in os.listdir(self.data_path + "test/"):
                print("Inferring subject %s" % subject_id)
                num_patches = np.load("%s/test/%s/nb_patches.npy" % (self.data_path, subject_id))
                X = np.empty((num_patches, self.sequence_length, self.len_patch))
                # Generate data
                for i in range(num_patches):
                    #load patches
                    X[i] = self.get_patches(np.load("%s/test/%s/%s.npy" % (self.data_path, subject_id, i)).reshape((*self.input_shape, self.num_channels)))
                y = self.model.predict(X)
                if self.num_classes == 2 :
                    #sigmoid function
                    y = 1 / (1 + np.exp(-y))
                    pred = np.mean(y[25:58], axis = 0)
                    writer.writerow({'subject_id' : subject_id, 'group' : np.around(pred)[0], "prob" : pred})
                else:
                    y= np.exp(y)/sum(np.exp(y))
                    pred = np.mean(y, axis = 0)
                    writer.writerow({'subject_id' : subject_id, 'group' : np.argmax(pred), "prob" : pred})
