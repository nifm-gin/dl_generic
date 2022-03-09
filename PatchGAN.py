import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, Concatenate, Conv3DTranspose, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout, Lambda
import os
from tensorflow.keras.initializers import RandomNormal
import numpy as np
from tensorflow.keras import Model, Input
import sys
import argparse
from Models import Model3D
from utils import dice_loss
import argparse
from patches_generation import find_indices
from ops import binary_ce
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

####################################################################################################

# PatchGAN model #

####################################################################################################
depth_2_window_size = {2 : 10,
                       3 : 22,
                       4 : 46}
class PatchGAN(Model3D):
    """
    PatchGAN model that heritates from Models class  
    Pix2pix discriminator
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 3, dropout=0., batch_norm=True,
                 initial_learning_rate=0.0002, batch_size = 8, model_type = "classification", padding='same', strides = 1, weights_path = None,
                 save_path = None, activation = "leakyRelu", depth = 3, features_factor = 1,
                 num_classes = 2, name = "Full_PatchGAN"):
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
        self.window_size = depth_2_window_size[self.depth]
        super().__init__(input_shape, data_path, num_channels, model, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation, save_path = save_path, name = name)
        #Load weights specified, separatly for D and G
        if weights_path is not None:
            try:
                assert os.path.isfile(weights_path), "Weights path specified does not exist, model kept as if"
                self.model.load_weights(weights_path)
                print("Models' weights loaded")
            except AssertionError as err:
                print(err)
            except Exception as err:
                print("Error raised while loading weights : %s.\n\tContinuing with model as if" % err)

        if self.model_type == "classification" :
            #choose on of them
            if self.num_classes == 2:
                self.model.compile(loss = binary_ce,
                                   optimizer = self.optimizer, metrics = ["accuracy"])
            else:
                self.model.compile(loss = binary_ce,
                                   optimizer = self.optimizer, metrics = ["accuracy"])
        elif self.model_type == "regression":
            self.model.compile(loss = "mse",
                               optimizer = self.optimizer, metrics = ["accuracy"])
        else :
            self.model.compile(loss = binary_ce,
                               optimizer = self.optimizer, metrics = ["accuracy"])
        
        #self.model.summary()

        
    def build_model(self):
        """
        Builds our classical CNN architecture
        Is adapted to being used as the discriminator of a cGAN => condition on model_type
        """
        #for this CNN with change initialization otherwise bound to be stuck in a local minimum
        #self.weights_init = RandomNormal(stddev = 0.02)
        patch_dim = [self.window_size for _ in self.input_shape]
        for i, d in enumerate(self.input_shape):
            if d < self.window_size:
                patch_dim[i] = d
        patch_dim = tuple(patch_dim)
        if len(self.input_shape) == 2:
            indices = find_indices((*patch_dim, 1), (*self.input_shape, 1), save_path = None)
        else:
            indices = find_indices(patch_dim, self.input_shape, save_path = None)
        assert len(indices) >= 1, "Something went wrong in patch-disc building, verify that input dimensions are compatible with patches size %s" % self.window_size
        gan_input = Input(shape=(*self.input_shape, self.num_channels), name="patchGAN_input0")
        list_gen_patch = []
        for indice in indices:
            if len(self.input_shape) == 2:
                x_patch = Lambda(lambda z: z[:, indice[0]:indice[1], indice[2]:indice[3], :])(gan_input)
            else:
                x_patch = Lambda(lambda z: z[:, indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5], :])(gan_input)
            list_gen_patch.append(x_patch)

        nb_filters = 64
        nb_conv = min(self.depth, int(np.floor(np.log(self.window_size) / np.log(2))))
        list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

        list_input = [Input(shape=(*patch_dim, self.num_channels), name="disc_input_%s" % i) for i in range(indices.shape[0])]
        #Defining a Disc for each patch
        #first conv
        x_input = Input(shape=(*patch_dim, self.num_channels), name="discriminator_input")
        x = self.conv(list_filters[0], self.filter_shape, strides=2, name="disc_conv_1", padding="same", kernel_initializer = self.weights_init)(x_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        
        # Next convs
        
        for i, f in enumerate(list_filters[1:]):
            name = "disc_conv_%s" % (i + 2)
            x = self.conv(f, self.filter_shape, strides=2, name=name, padding="same", kernel_initializer = self.weights_init)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        x_flat = Flatten()(x)
        #x = Dense(2, activation="sigmoid", name="disc_dense", kernel_initializer = self.weights_init)(x_flat)
        x = Dense(1, name="disc_dense", kernel_initializer = self.weights_init)(x_flat)
        PatchGAN = Model(inputs=[x_input], outputs=[x], name="PatchGAN")
        #build the discriminator => CNN
        x = [PatchGAN(patch) for patch in list_gen_patch]
        #self.model = Model(inputs = gan_input, outputs = x, name = self.name)
        
        if len(x) > 1:
            x = Concatenate()(x)
        else:
            print(x)
            x = x[0]
        #x_out = Dense(2, activation="sigmoid", name="disc_output", kernel_initializer = self.weights_init)(x)
        x_out = Dense(1, name="disc_output", kernel_initializer = self.weights_init)(x)
        self.model = Model(inputs = gan_input, outputs = x_out, name = self.name)
        
       
        
##################################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments for patches creation.')
    parser.add_argument("-is", '--input_shape', type=int, nargs='+', default = (64, 64, 64),
                        help='Input Shape of model, by default 64*64*64')
    parser.add_argument("-dp", '--data_path', type=str, default = None,
                        help = "path to the directory containing the patches")
    parser.add_argument("-d", '--depth', type = int, default = 4,
                        help='Depth of our Unet model, by default 4')
    parser.add_argument("-bs", '--batch_size', type = int, default = 16,
                        help='Batch size required for training, by default 8')
    parser.add_argument("-lr", '--learning_rate', type = float, default = 0.001,
                        help='Learning rate required to start our training, 10E-3 by default.')
    parser.add_argument("-s", '--strides', type = int, nargs = '+', default = 1,
                        help='Strides used for each convolutional layer, by default (1, 1, 1)')
    parser.add_argument("-ps", '--pool_size', type = int, nargs = '+', default = 2,
                        help='Size of each max pool layer, by default (2, 2, 2)')
    parser.add_argument("-fs", '--filter_size', type = int, nargs = '+', default = 3,
                        help='Depth of our Unet model, by default (3,3,3)')
    parser.add_argument("-nc", '--num_channels', type = int, default = 1,
                        help='Number of contrasts as input, by default 1')
    parser.add_argument("-e", '--epochs', type = int, default = 10,
                        help='Number of training epochs , by default 10')
    parser.add_argument("-ff", '--features_factor', type = int, default = 8,
                        help='Factor by which multiply the powers of 2, will give the features number needed at each depth level , by default 8')
    parser.add_argument("-spe", '--steps_per_epoch', type = int, default = 25,
                        help='Number of steps for each epoch , by default 25')
    parser.add_argument("-wp", '--weights_path', type=str, default = None,
                        help = "path to the file containing already trained model's weighs. None by default")
    parser.add_argument("-sp", '--save_path', type=str, default = "./models/",
                        help = "path to the directory in which to save our models weights.")
    parser.add_argument("-mt", '--model_type', type=str, default = "classification",
                        help = "Output type of our model, 'generation' by default")
    parser.add_argument('--infer_only', action = 'store_true',
                        help = "Indicates wether we want to train or not our model")
    parser.add_argument('--train_only', action = 'store_true',
                        help = "Indicates wether we want to infer or not our model")
    parser.add_argument('--batch_norm', action = 'store_true',
                        help = "Will add batchnormalization layers after each conv layer of encoder")
    parser.add_argument('--gpu', type = int, nargs = '+', default = [],
                        help='Which gpu are required, None by default')

    
    args = parser.parse_args()
    if len(args.gpu) == 1 :
        os.environ["CUDA_VISIBLE_DEVICES"]="%s" % args.gpu[0]
    elif args.gpu == 2:
        os.environ["CUDA_VISIBLE_DEVICES"]="%s,%s" % tuple(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=""

