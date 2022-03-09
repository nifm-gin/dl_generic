import tensorflow as tf
# if __name__ == "__main__":
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     try:
#         config = tf.config.experimental.set_memory_growth(physical_devices[1], True)
#     except:
#         print("Only one GPU found")
from tensorflow.keras.layers import Concatenate, LeakyReLU, Flatten, Dense, Dropout, Lambda
import os
import time
from tensorflow.keras.initializers import TruncatedNormal, GlorotNormal, GlorotUniform
import numpy as np
from tensorflow.keras import Model, Input
import sys
import argparse
import datetime
from Models import Model3D
from ops import binary_ce, calc_l1_loss, CustomSchedule, masked_mse_loss
from layers import TransformerDecoder, TransformerEncoder
import csv
import argparse
from patches_generation import find_indices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152


####################################################################################################
 
# CNN model #

####################################################################################################

class VisionTransformer(Model3D):
    """
    CNN model that heritates from Models class  
    
    """
    def __init__(self, input_shape, data_path, patch_shape = 16, num_channels = 1, model = None, dropout=0.4, d_model = 512, vocabulary_size = 10000,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "generation", weights_path = None, num_heads = 8,
                 save_path = None, activation = "leakyRelu", depth = 4, features_factor = 3072, name = "imageTransfo"):
        """
        parameters:
        ----------
        depth : int,
            depth of our model, number of conv block (and deconv block) in the encoder (decoder)
        """
        #check dimension of model :
        self.features_factor = features_factor
        self.depth = depth
        self.dropout = dropout
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
        #get patches
        if len(input_shape) == 2:
            self.indices = find_indices((*self.patch_shape, 1), (*input_shape, 1), save_path = None)
        else:
            self.indices = find_indices(self.patch_shape, input_shape, save_path = None)
       
        assert len(self.indices) >= 1, "Something went wrong in patch-disc building, verify that input dimensions are compatible with patches size %s" % self.window_size
        self.sequence_length = len(self.indices)
        super().__init__(input_shape, data_path, num_channels, model, 0, 0, dropout, False, initial_learning_rate, batch_size, model_type, None, 0, weights_path, activation, save_path = save_path, name = name)
        #self.input_shape = (self.sequence_length, self.patch_shape)
        self.model.summary()

        #optimizer
        self.learning_rate = CustomSchedule(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5, beta_2=0.98, epsilon=1e-9)
        
        
    def build_model(self):
        """
        Builds an image transformer architecture
        """
        input_patches = Input((self.sequence_length, self.len_patch))
        target_patches = Input((self.sequence_length + 1, self.len_patch))

        self.encoder = TransformerEncoder(self.depth, self.d_model, self.num_heads, self.dff, self.embedding, self.sequence_length)
        self.decoder = TransformerDecoder(self.depth, self.d_model, self.num_heads, self.dff, self.embedding, self.sequence_length + 1)
            
        x = self.encoder(input_patches, training = True)
        x, attention_weights = self.decoder(target_patches, x, self.look_ahead_mask(), training = True)
        if self.d_model < self.len_patch:
            #Embedding was used so 
            x = Dense(self.len_patch, activation = 'tanh')(x)
        else:
            #No embedding was used here, so return just activated output
            x =  tf.keras.activations.tanh(x)
            
        self.model = Model(inputs = [input_patches, target_patches], outputs = [x, attention_weights], name = self.name)

    def look_ahead_mask(self):
        """
        returns decoder lookahead mask. Decoder has access to all encoder output its previous prediction only  
        So it can predict next patch
        """
        mask =  1 - tf.linalg.band_part(tf.ones((self.sequence_length, self.sequence_length)), -1, 0)
        #tf.print(mask)
        return mask  # (seq_len, seq_len)
    
    @tf.function
    def train_step(self, inp, tar, epoch, write = False):
        # The target is divided into tar_inp and tar_real.
        #tar_inp is passed as an input to the decoder.
        #tar_real is that same input shifted by 1
        #At each location in tar_input, tar_real contains the next token that should be predicted.

        tar_inp = tar[:,:-1]
        tar_real = tar[:, 1:]
        #Shifting input and target doesnt make sense here as our images always have the same dimensions
        with tf.GradientTape() as tape:
            predictions, _ = self.model([inp, tar_inp], training = True)
            loss = calc_l1_loss(tar_real, predictions) + masked_mse_loss(tar_real, predictions)
            #loss = mse_loss(tar_real, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        return {'L1_loss' : self.train_loss.result()}
                
    @tf.function
    def val_step(self, inp, tar, epoch, on_cpu):
        with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
            predictions, _ = self.model(inp, training = False)
            loss = calc_l1_loss(tar, predictions)
            loss = tf.reduce_mean(tf.abs(tf.boolean_mask(real_image, mask) - tf.boolean_mask(cycled_image, mask)))
            return{'L1_val_loss' : loss}
    
    def get_patches(self, X):
        patches = []
        for indice in self.indices:
            if len(self.input_shape) == 2:
                patch = X[indice[0]:indice[1], indice[2]:indice[3], :]
            else:
                patch = X[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5], :]
            patches.append(patch.flatten())
        return patches
    
    def data_generator(self, data_type = 'train'):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        X = np.empty((self.batch_size if data_type == "train" else num_patches, self.sequence_length, self.len_patch), dtype=np.float32)
        y = -np.ones((self.batch_size if data_type == "train" else num_patches, self.sequence_length + 2, self.len_patch), dtype=np.float32)
        if data_type == "train":
            indices = np.random.permutation(int(num_patches))
            #for cpt in range(num_patches // self.batch_size):
            #indexes = indices[cpt * self.batch_size : min((cpt + 1) * self.batch_size, num_patches)]
            while True:
                indexes = np.random.choice(num_patches, self.batch_size)
                for i, ID in enumerate(indexes):
                    #load patches
                    X[i] = self.get_patches(np.load("%s/train/%s.npy" % (self.data_path, ID)).reshape((*self.input_shape, self.num_channels)))
                    y[i, 1:-1] = self.get_patches(np.load("%s/train/output_%s.npy" % (self.data_path, ID)).reshape((*self.input_shape, 1)))
                yield X, y          
            
        else:
            for i in range(num_patches):
                X[i] = self.get_patches(np.load("%s/val/%s.npy" % (self.data_path, i)).reshape((*self.input_shape, self.num_channels)))
                y[i] = self.get_patches(np.load("%s/val/output_%s.npy" % (self.data_path, i)).reshape((*self.input_shape, 1)))
                # X = X.reshape((self.batch_size, self.sequence_length, self.len_patch))
                # y = y.reshape((self.batch_size, self.sequence_length, self.len_patch))
            return X, y

    def infer(self):
        """
        method infering test data with our model
        Infering data present in test directory, subject wisely
        """
        if not os.path.exists("%s/infered/" % (self.save_path)):
            os.makedirs("%s/infered/" % (self.save_path))
        csv_name = "%s/infered/predictions.csv" % self.save_path
        for subject_id in os.listdir(self.data_path + "test/"):
            print("Inferring subject %s" % subject_id)
            num_patches = np.load("%s/test/%s/nb_patches.npy" % (self.data_path, subject_id))
            X = np.empty((num_patches, self.sequence_length, self.len_patch * self.num_channels))
            y = np.zeros((num_patches, self.sequence_length + 1, self.len_patch * self.num_channels))
            final_y = np.empty((num_patches, *self.input_shape, 1))
            # Generate data
            for i in range(num_patches):
                #load patches
                X[i] = self.get_patches(np.load("%s/test/%s/%s.npy" % (self.data_path, subject_id, i)).reshape((*self.input_shape, self.num_channels)))
            for pos in range(self.sequence_length):
                preds, _ = self.model.predict([X,  y])
                preds[:, pos][X[:,pos] < -0.995] = -1
                y[:,pos + 1,:] = preds[:, pos,:]
            # y, _ = self.model.predict([X,  X])
            # y = y.reshape((-1, self.sequence_length, self.len_patch * self.num_channels))
            #y[X < -0.995] = -1
            y = y[:, 1:]
            #print("MAE : ", np.mean(np.abs(y - X)))
            #Once we have our predicted sequence we want to turn it into our final image
            for pos in range(self.sequence_length):
                patch =y[:, pos]
                indice = self.indices[pos]
                if len(self.input_shape) == 2:
                    final_y[:, indice[0]:indice[1], indice[2]:indice[3], :] = patch.reshape((patch.shape[0], *self.patch_shape, 1))
                else:
                    final_y[:, indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5], :] = patch.reshape((patch.shape[0], *self.patch_shape, 1))
                                
            if not os.path.exists("%s/infered/%s/" % (self.save_path, subject_id)):
                os.makedirs("%s/infered/%s/" % (self.save_path, subject_id))
            for i, pred in enumerate(final_y):
                np.save("%s/infered/%s/output_%s.npy" % (self.save_path, subject_id, i), pred)

             
