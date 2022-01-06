import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, Concatenate, Conv3DTranspose, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout
import os
from tensorflow.keras.initializers import TruncatedNormal, GlorotNormal, GlorotUniform
import numpy as np
from tensorflow.keras import Model, Input
import sys
from Models import Model3D
from ops import binary_ce, categorical_ce
from sn import SpectralNormalization
import csv
import pandas as pd
####################################################################################################

# CNN model #

####################################################################################################

class CNN(Model3D):
    """
    CNN model that heritates from Models class  
    
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 3, dropout=0.4, batch_norm=True,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "classification", padding='same', strides = 1, weights_path = None,
                 save_path = None, activation = "leakyRelu", depth = 4, features_factor = 16, num_classes = 1, name = "CNN", class_weights = None,
                 additional_inputs = None, data_augmentation = [], gt_path = None):

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
            self.maxPool = MaxPool3D
        else:
            #2D model
            self.conv = Conv2D
            self.maxPool = MaxPool2D
        self.features_factor = features_factor
        self.depth = depth
        self.dropout = dropout
        self.num_classes = num_classes
        self.additional_inputs = additional_inputs
        super().__init__(input_shape, data_path, num_channels, model, pool_size, filter_shape, dropout, batch_norm,
                         initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation,
                         save_path = save_path, name = name, decay_rate = "exponential", data_augmentation = data_augmentation)

        self.class_weights = class_weights
        self.gt_path = gt_path
        self.model.summary()
        if self.model_type == "classification" or self.model_type == "discrimination":
            #choose on of them
            if self.num_classes == 2:
                self.loss = binary_ce
            else:
                self.loss = categorical_ce
        elif self.model_type == "regression":
            self.loss = tf.keras.losses.mean_squared_error
        else :
            self.loss = binary_ce
    def build_model(self):
        """
        Builds our classical CNN architecture
        Is adapted to being used as the discriminator of a cGAN => condition on model_type
        """
        #for this CNN with change initialization otherwise bound to be stuck in a local minimum
        self.weights_init = TruncatedNormal(0., 0.02)
        #self.weights_init = GlorotNormal()
        #self.weights_init = GlorotUniform()
        
        inputs = Input(self.input_shape + (self.num_channels,))
        ff = self.features_factor
        if self.model_type == "discrimination":
            if self.conv == Conv3D:
                x = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(inputs)
                x = BatchNormalization()(x)
            else:
                #x = SpectralNormalization(self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init))(inputs)
                x = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(inputs)
                x = BatchNormalization()(x)
        else:
            x = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(inputs)
            if self.batch_norm:
                x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if self.model_type != "discrimination":
            x = self.maxPool(self.pool_size)(x)
            
        for d in range(1, self.depth):
            if self.model_type == "discrimination":
                x = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(x)
                x = BatchNormalization()(x)
                #x = SpectralNormalization(self.conv(ff * 2**d, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init))(x)
            else:
                x = self.conv(ff * 2**d, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(x)
                if self.batch_norm:
                    x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.model_type != "discrimination":
                x = self.maxPool(pool_size = self.pool_size)(x)
        if self.model_type != "discrimination":
            #Second last part of our model
            x = Dropout(self.dropout)(x)
            features = Flatten()(x)
            x = Dense(128, kernel_initializer = self.weights_init)(features)
            x = LeakyReLU()(x)
            if self.additional_inputs is not None :
                clinical_inputs = Input(len(self.additional_inputs))
                x = Concatenate()([x, clinical_inputs])
                inputs = [inputs, clinical_inputs]
            if self.num_classes == 2:
                x = Dense(1, activation='linear', kernel_initializer = self.weights_init)(x)
            else:
                #classification with n_class > 2
                x = Dense(self.num_classes, activation='linear', kernel_initializer = self.weights_init)(x)
            self.features_extractor = Model(inputs = inputs, outputs = features, name = self.name + "_extractor")
        else:
            #Discriminator linear activation as we compute loss on logits
            x = self.conv(1, self.filter_shape, padding='same',
                          kernel_initializer=self.weights_init)(x)
        self.model = Model(inputs = inputs, outputs = x, name = self.name)
        
    def generator_val(self, data_type = 'val'):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        X = np.empty((num_patches, *self.input_shape, self.num_channels))
        if self.additional_inputs is not None:
            X = [X, np.empty((num_patches, len(self.additional_inputs)))]
        if self.num_classes == 2:
            y = np.zeros((num_patches, 1))
        else:
            y = np.zeros((num_patches, self.num_classes))
        for i in range(num_patches):
            data = np.load("%s/%s/%s.npz" % (self.data_path, data_type, i))
            if self.additional_inputs is not None:
                X[0][i] = data["data"].reshape((*self.input_shape, self.num_channels))
                X[1][i] = data["add_in"]
            else:
                X[i] = data["data"].reshape((*self.input_shape, self.num_channels))
            if self.num_classes == 2:
                y[i] = data["group"][0]
            else:
                y[i, data["group"]][1] = 1
        return X, y
    
    def data_generator(self, data_type = 'train'):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        while True:
            indexes = np.random.choice(num_patches, self.batch_size)
            X = np.empty((self.batch_size, *self.input_shape, self.num_channels))
            if self.additional_inputs is not None:
                X = [X, np.empty((self.batch_size, len(self.additional_inputs)))]
            if self.num_classes == 2:
                y = np.zeros((self.batch_size, 1))
            else:
                y = np.zeros((self.batch_size, self.num_classes))
                #y = np.random.randint(2, size = (self.batch_size, 1))
            
            # Generate data
            for i, ID in enumerate(indexes):
                 #load patches
                data = np.load("%s/%s/%s.npz" % (self.data_path, data_type, ID))
                if self.additional_inputs is not None:
                    X[0][i] = data["data"].reshape((*self.input_shape, self.num_channels))
                    X[1][i] = data["add_in"]
                else:
                        X[i] = data["data"].reshape((*self.input_shape, self.num_channels))
                if self.num_classes == 2:
                    y[i] = data["group"][0]
                else:
                    y[i, data["group"][0]] = 1
            for augmentation in self.augmentations:
                for i in range(self.batch_size):
                    if np.random.uniform() < (0.5 / len(self.augmentations)):
                        X[i] = augmentation(X[i])
            yield X, y

    def infer(self, saliency = False):
        """
        method infering test data with our model
        Infering data present in test directory, subject wisely
        """
        if self.gt_path is not None:
            ground_truth = pd.DataFrame(pd.read_csv(self.gt_path)).to_numpy()
        if not os.path.exists("%s/infered/" % (self.save_path)):
            os.makedirs("%s/infered/" % (self.save_path))
        csv_name = "%s/infered/predictions.csv" % self.save_path
        csv_detailed_name = "%s/infered/predictions_by_slice.csv" % self.save_path
        with open(csv_detailed_name, 'w', newline='') as details_file:
            with open(csv_name, 'w', newline='') as csvfile:
                if self.gt_path is not None:
                    fieldnames = ['subject_id', 'group', 'ground_truth', "slice", "prob"]
                else :
                    fieldnames = ['subject_id', 'group', "slice", "prob"]
                writer1 = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer1.writeheader()
                writer2 = csv.DictWriter(details_file, fieldnames=fieldnames)
                writer2.writeheader()
                for subject_id in os.listdir(self.data_path + "test/"):
                    print("Inferring subject %s" % subject_id)
                    num_patches = np.load("%s/test/%s/nb_patches.npy" % (self.data_path, subject_id))
                    X = np.empty((num_patches, *self.input_shape, self.num_channels))
                    if self.additional_inputs is not None:
                        X = [X, np.empty((num_patches, len(self.additional_inputs)))]
                        # Generate data
                    for i in range(num_patches):
                        #load patches
                        if self.additional_inputs is not None:
                            data = np.load("%s/test/%s/%s.npz" % (self.data_path, subject_id, i))
                            X[0][i] = data["data"].reshape((*self.input_shape, self.num_channels))
                            X[1][i] = data["add_in"]
                        else:
                            X[i] = np.load("%s/test/%s/%s.npz" % (self.data_path, subject_id, i))["data"].reshape((*self.input_shape, self.num_channels))
                    y = self.model.predict(X)
                    if self.model_type != "discrimination":
                        if self.gt_path is not None:
                            gt_patient = ground_truth[np.where(ground_truth == [subject_id])[0][0], 0]
                            if self.num_classes == 2 :
                                #sigmoid function
                                y = 1 / (1 + np.exp(-y))
                                pred = np.mean(y, axis = 0)
                                #print(pred, np.around(pred))
                                writer1.writerow({'subject_id' : subject_id, 'group' : np.around(pred[0]), 'ground_truth' : gt_patient, "slice" :"null", "prob" : pred[0]})
                                for num_slice, pred in enumerate(y):
                                    writer2.writerow({'subject_id' : subject_id, 'group' : np.around(pred[0]), 'ground_truth' : gt_patient, "slice" : num_slice, "prob" : pred[0]})

                            else:
                                pred = np.mean(y, axis = 0)
                                writer1.writerow({'subject_id' : subject_id, 'group' : np.argmax(pred), 'ground_truth' : gt_patient, "slice" :"null", "prob" : pred})
                        else:
                            if self.num_classes == 2 :
                                #sigmoid function
                                y = 1 / (1 + np.exp(-y))
                                pred = np.mean(y, axis = 0)
                                #print(pred, np.around(pred))
                                writer1.writerow({'subject_id' : subject_id, 'group' : np.around(pred[0]), "slice" :"null", "prob" : pred[0]})
                                for num_slice, pred in enumerate(y):
                                    writer2.writerow({'subject_id' : subject_id, 'group' : np.around(pred[0]), "slice" : num_slice, "prob" : pred[0]})

                            else:
                                pred = np.mean(y, axis = 0)
                                writer1.writerow({'subject_id' : subject_id, 'group' : np.argmax(pred), "slice" :"null", "prob" : pred})
                    else:
                        if not os.path.exists("%s/infered/%s/" % (self.save_path, subject_id)):
                            os.makedirs("%s/infered/%s/" % (self.save_path, subject_id))
                        for i, pred in enumerate(y):
                            np.save("%s/infered/%s/output_%s.npy" % (self.save_path, subject_id, i), pred)
                    if not os.path.exists("%s/infered/%s/" % (self.save_path, subject_id)):
                        os.makedirs("%s/infered/%s/" % (self.save_path, subject_id))
                    if saliency:
                        saliency = self.saliency(X)
                        for i, patch in enumerate(saliency):
                            np.save("%s/infered/%s/saliency_%s.npy" % (self.save_path, subject_id, i), patch)

    
