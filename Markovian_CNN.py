import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, Concatenate, Conv3DTranspose, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout
import os
from tensorflow.keras.initializers import TruncatedNormal
import numpy as np
from tensorflow.keras import Model, Input
import sys
import argparse
from Models import Model3D
from utils import dice_loss
import argparse
from ops import celoss
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

####################################################################################################

# CNN model #

####################################################################################################

class MarkovianCNN(Model3D):
    """
    CNN model that heritates from Models class  
    
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 3, dropout=0., batch_norm=True,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "classification", padding='valid', strides = 1, weights_path = None,
                 save_path = None, activation = "leakyRelu", depth = 4, features_factor = 1, num_classes = 2, name = "Mark"):
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
        super().__init__(input_shape, data_path, num_channels, model, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation, save_path = save_path, name = name)
        self.model.compile(loss = celoss,
                           optimizer = self.optimizer, metrics = ["accuracy"])

    def build_model(self):
        """
        Builds our classical CNN architecture
        Is adapted to being used as the discriminator of a cGAN => condition on model_type
        """
        #for this CNN with change initialization otherwise bound to be stuck in a local minimum
        self.weights_init = TruncatedNormal(0., 0.2)
        inputs = Input(self.input_shape + (self.num_channels,))
        ff = self.features_factor
        x = self.conv(ff, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(inputs)
        x = LeakyReLU()(x)
        for d in range(1, self.depth - 1):
            x = self.conv(ff * 2**d, self.filter_shape, strides = self.strides, padding = self.padding, kernel_initializer = self.weights_init)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        #Second last part of our model
        x = self.conv(ff * 2**self.depth - 1, 1, padding = self.padding, kernel_initializer = self.weights_init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = self.conv(2, 1, padding = self.padding, kernel_initializer = self.weights_init)(x)
        self.model = Model(inputs = inputs, outputs = x, name = self.name)
        
    def generator_val(self, data_type = 'val'):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        X = np.empty((num_patches, *self.input_shape, self.num_channels))
        y = np.zeros((num_patches, 1))
        if data_type == 'val':
            for i in range(num_patches):
                X[i] = np.load("%s/%s/%s.npy" % (self.data_path, data_type, i)).reshape((*self.input_shape, self.num_channels))
                if i < num_patches // 2 :
                    y[i] = 1
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
            y = np.zeros((self.batch_size, 1))
            #y = np.random.randint(2, size = (self.batch_size, 1))
            
            # Generate data
            for i, ID in enumerate(indexes):
                 #load patches
                X[i] = np.load("%s/%s/%s.npy" % (self.data_path, data_type, ID)).reshape((*self.input_shape, self.num_channels))
                if ID < num_patches // 2:
                    y[i] = 1
            yield X, y
       
        
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

    model = CNN(input_shape = tuple(args.input_shape), data_path = args.data_path,
                 num_channels = args.num_channels, batch_norm = args.batch_norm,
                 pool_size = args.pool_size, filter_shape = args.filter_size,
                 batch_size = args.batch_size, initial_learning_rate = args.learning_rate,
                 strides = args.strides, depth = args.depth, weights_path = args.weights_path,
                 save_path = args.save_path, model_type = args.model_type, features_factor = args.features_factor)
    if not args.infer_only:
        model.train(args.epochs, args.steps_per_epoch)
        model.model.load_weights(model.save_path + "/best.h5")
        model.model.save_weights(model.save_path + '/final_model_weights.h5')
        print("Models weights saved.")
    if not args.train_only:
        model.infer()
        print("Inference done!")
    print(model.model.predict(model.generator_val()[0]))
    print("Exiting script")
