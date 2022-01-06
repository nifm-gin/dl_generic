import sys
import time
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
from Markovian_CNN import MarkovianCNN
from PatchGAN import PatchGAN
sys.path.append("/home/cackowss/dl_generic/utils/")
from ops import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
####################################################################################################

# DualGAN model #
#inpired from https://github.com/duxingren14/DualGAN

####################################################################################################
    
class DualGAN(Model3D):
    """
    Unet model that heritates from Models class  
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 4, dropout=.1, batch_norm=False,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "dualGAN", padding='same', strides = 2,
                 weights_path_disc = None, weights_path_gen_A = None, weights_path_gen_B = None, weights_path = None,
                 save_path = None, activation = "relu", depth = 4, features_factor = 8, optimizer = "adam", name = "DualGAN"):
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
        self.generator_A, self.generator_B, self.discriminator_A, self.discriminator_B = None, None, None, None
        self.depth = depth
        #Two optimizer objects for Gen and Disc
        super().__init__(input_shape, data_path, num_channels, None, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation, save_path = save_path, name = name)
        #Use a exponential decay learning rate
        # learning_rate = tf.train.exponential_decay(self.initial_learning_rate, tf.Variable(0, trainable = False), 
        #                                            5, 0.8, staircase=True)
        #Use several optimizer for each training
        self.generator_A_optimizer = tf.keras.optimizers.Adam(lr = self.initial_learning_rate, beta_1=0.5)
        self.generator_B_optimizer = tf.keras.optimizers.Adam(lr = self.initial_learning_rate, beta_1=0.5)
        self.discriminator_A_optimizer = tf.keras.optimizers.Adam(lr = self.initial_learning_rate * 2, beta_1=0.5)
        self.discriminator_B_optimizer = tf.keras.optimizers.Adam(lr = self.initial_learning_rate * 2, beta_1=0.5)
        self.optimizer = [self.generator_A_optimizer, self.generator_B_optimizer, self.discriminator_A_optimizer, self.discriminator_B_optimizer]
        self.scheduler = Scheduler(self.optimizer,
                                      ["Gen_A", "Gen_B", "Disc_A", "Disc_B"],
                                      ratios = [1, 1, 2, 2],
                                      patience = 5, early_stopping = 30)

        #Load weights specified, separatly for D and G
        if weights_path_disc is not None:
            try:
                assert os.path.isfile(weights_path_disc), "Weights path specified for discriminator does not exist, model kept as if"
                self.discriminator_A.load_weights(weights_path_disc)
                self.discriminator_B.load_weights(weights_path_disc)
                print("Discriminators' weights loaded")
            except AssertionError as err:
                print(err)
            except Exception as err:
                print("Error raised while loading discriminator weights : %s.\n\tContinuing with model as if" % err)
        if weights_path_gen_A is not None and weights_path_gen_B is not None:
            try:
                assert os.path.isfile(weights_path_gen_A) and os.path.isfile(weights_path_gen_B), "Weights path specified for generators do not exist, models kept as if"
                self.generator_A.load_weights(weights_path_gen_A)
                self.generator_B.load_weights(weights_path_gen_B)
                print("Generators' weights loaded")
            except AssertionError as err:
                print(err)
            except Exception as err:
                print("Error raised while loading generator weights : %s.\n\tContinuing with model as if" % err)
        print("Init done")
        self.generator_A.summary()
        self.discriminator_B.summary()
        self.model.summary()
        #tf.keras.utils.plot_model(self.model, to_file = "dualGAN.png", show_shapes=True, dpi=64)
        
    def build_model(self):
        #Build both generators => Unet
        self.generator_A = Unet(self.input_shape, self.data_path, self.num_channels, None, self.pool_size,
                                4, self.dropout, self.batch_norm, self.initial_learning_rate,
                                self.batch_size, "harmonization", self.padding, self.strides, None,
                                self.save_path, self.activation, self.depth, self.features_factor,
                                attention = False, name = "Unet_A").model
        
        self.generator_B = Unet(self.input_shape, self.data_path, self.num_channels, None, self.pool_size,
                                4, self.dropout, self.batch_norm, self.initial_learning_rate,
                                self.batch_size, "harmonization", self.padding, self.strides, None,
                                self.save_path, self.activation, self.depth, self.features_factor,
                                attention = False, name = "Unet_B").model
        genA_inputs = Input(self.input_shape + (self.num_channels,))
        #each generator tries to map the negative-residual 
        generated_image_A = self.generator_A(genA_inputs)

        genB_inputs = Input(self.input_shape + (self.num_channels,))
        generated_image_B = self.generator_B(genB_inputs)
        #Building discriminator :
        if self.model_type == "dualGAN":
            self.discriminator_A = CNN(self.input_shape, self.data_path, 1, None, self.pool_size,
                                       self.filter_shape, self.dropout, self.batch_norm, self.initial_learning_rate * 2,
                                       self.batch_size, "discrimination", self.padding, self.strides, None,
                                       self.save_path, self.activation, 3, 64, 2, "CNN_A").model
            self.discriminator_B = CNN(self.input_shape, self.data_path, 1, None, self.pool_size,
                                       self.filter_shape, self.dropout, self.batch_norm, self.initial_learning_rate * 2,
                                       self.batch_size, "discrimination", self.padding, self.strides, None,
                                       self.save_path, self.activation, 3, 64, 2, 'CNN_B').model
            
        elif self.model_type == "dualGAN_mark":
            self.discriminator_A = MarkovianCNN(self.input_shape, self.data_path, 1, None, self.pool_size,
                                                self.filter_shape, self.dropout, self.batch_norm, 0.0002,
                                                self.batch_size, "classification", "valid", self.strides, None,
                                                self.save_path, self.activation, 3, self.features_factor, 2, name = "mark_A").model
            self.discriminator_B = MarkovianCNN(self.input_shape, self.data_path, 1, None, self.pool_size,
                                                self.filter_shape, self.dropout, self.batch_norm, 0.0002,
                                                self.batch_size, "classification", "valid", self.strides, None,
                                                self.save_path, self.activation, 3, self.features_factor, 2, "mark_B").model
        else:
            self.discriminator_A = PatchGAN(self.input_shape, self.data_path, 1,
                                            None, self.pool_size,
                                            self.filter_shape, self.dropout, self.batch_norm, 0.0002,
                                            self.batch_size, "classification", self.padding, self.strides, None,
                                            self.save_path, self.activation, 4, self.features_factor, 2,
                                            name = "PatchDisc_A").model
            self.discriminator_B = PatchGAN(self.input_shape, self.data_path, 1,
                                            None, self.pool_size,
                                            self.filter_shape, self.dropout, self.batch_norm, 0.0002,
                                            self.batch_size, "classification", self.padding, self.strides, None,
                                            self.save_path, self.activation, 4, self.features_factor, 2,
                                            name = "PatchDisc_B").model

        recov_A = self.generator_A(generated_image_B)
        recov_B = self.generator_B(generated_image_A)
        discA_output = self.discriminator_A(generated_image_A)
        discB_output = self.discriminator_B(generated_image_B)
        self.model = Model(inputs = [genA_inputs, genB_inputs], outputs = [discA_output, discB_output, recov_A, recov_B], name = self.name)
        
    def data_generator(self, data_type = "train"):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = {"A" : np.load("%s/A/%s/nb_patches.npy" % (self.data_path, data_type)),
                       "B" : np.load("%s/B/%s/nb_patches.npy" % (self.data_path, data_type))}
        max_num_patches = max(num_patches['A'], num_patches['B'])
        min_num_patches = min(num_patches['A'], num_patches['B'])
        
        X = np.empty((self.batch_size if data_type == "train" else min_num_patches, *self.input_shape, self.num_channels), dtype = np.float32)
        X2 = np.empty((self.batch_size if data_type == "train" else min_num_patches, *self.input_shape, self.num_channels), dtype = np.float32)
        while True:
            if data_type == "train":
                indexes = np.random.choice(min_num_patches, self.batch_size)
                # Generate data
                for i, ID in enumerate(indexes):
                    #load patches
                    X[i] = np.load("%s/B/train/%s.npy" % (self.data_path, ID % num_patches["B"])).reshape((*self.input_shape, self.num_channels))
                    X2[i] = np.load("%s/A/train/%s.npy" % (self.data_path, ID % num_patches["A"])).reshape((*self.input_shape, self.num_channels))

            else:
                #validation data
                for i in range(min_num_patches):
                    #load patches
                    X[i] = np.load("%s/B/val/%s.npy" % (self.data_path, i)).reshape((*self.input_shape, self.num_channels))
                    X2[i] = np.load("%s/A/val/%s.npy" % (self.data_path, i)).reshape((*self.input_shape, self.num_channels))
                    
                indices = np.random.choice(X.shape[0], 400)
                X, X2 = X[indices], X2[indices]
            yield X2, X
       
    def train(self, epochs = 10, steps = 25, load_ckpt = False, val_on_cpu = False):
        """
        method to train our model
        """ 
        checkpoint_path = self.save_path + "/checkpoints/train"
        ckpt = tf.train.Checkpoint(generator_A=self.generator_A,
                                   generator_B=self.generator_B,
                                   discriminator_A=self.discriminator_A,
                                   discriminator_B=self.discriminator_B,
                                   generator_A_optimizer=self.generator_A_optimizer,
                                   generator_B_optimizer=self.generator_B_optimizer,
                                   discriminator_A_optimizer=self.discriminator_A_optimizer,
                                   discriminator_B_optimizer=self.discriminator_B_optimizer)
        
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        
        # if a checkpoint exists, restore the latest checkpoint.
        try:
            if ckpt_manager.latest_checkpoint and load_ckpt:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                print ('Latest checkpoint restored!!')
                for opt in self.optimizer:
                    opt.lr.assign(self.initial_learning_rate)
        except:
            print("Could not restore latest checkpoint, continuing as if!")
        print("Start training")
        for A_val, B_val in self.data_generator("val"):
            #loading validation data only once
            for e in range(epochs):
                start = time.time()
                n = 1
                for image_a, image_b in self.data_generator():
                    train_step_dualGAN(self, image_a, image_b, tf.constant(e, dtype = tf.int64), write = (n % steps) == 0)
                    if n % steps == 0:
                        print ('.', end='')
                        break
                    n+=1
                #Validation step
                if (e + 1) % 12 == 0:
                    val_losses = val_step_dualGAN(self, A_val, B_val, tf.constant(e, dtype = tf.int64), val_on_cpu)
                    #print("Val loss : %s " % val_losses[0].numpy())
                    if self.scheduler.update(val_losses):
                        ckpt_save_path = ckpt_manager.save()
                        print ('Saving checkpoint for epoch {} at {}'.format(e+1,
                                                                             ckpt_save_path))
                print ('Time taken for epoch {} is {} sec\n'.format(e + 1,
                                                                    time.time()-start))
            break
        #Save all model and different Gen & Disc
        self.discriminator_A.save_weights(self.save_path + '/best_d_A.h5')
        self.generator_A.save_weights(self.save_path + '/best_g_A.h5')
        self.discriminator_B.save_weights(self.save_path + '/best_d_B.h5')
        self.generator_B.save_weights(self.save_path + '/best_g_B.h5')
        #self.model.save_weights(self.save_path + '/best.h5')
            
    def infer(self, load_ckpt = False, saliency = False):
        """
        method infering test data with our model
        Infering data present in test directory, subject wisely
        Scans will go through one generator only as we aim to harmonize a site to the other
        """
        checkpoint_path = self.save_path + "/checkpoints/train"
        ckpt = tf.train.Checkpoint(generator_A=self.generator_A,
                                   generator_B=self.generator_B,
                                   discriminator_A=self.discriminator_A,
                                   discriminator_B=self.discriminator_B,
                                   generator_A_optimizer=self.generator_A_optimizer,
                                   generator_B_optimizer=self.generator_B_optimizer,
                                   discriminator_A_optimizer=self.discriminator_A_optimizer,
                                   discriminator_B_optimizer=self.discriminator_B_optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        try:
            if ckpt_manager.latest_checkpoint and load_ckpt:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                print ('Latest checkpoint restored!!')
        except Exception as err:
            print("Could not restore latest checkpoint, continuing as if! \n\t", err)

        for site in ["A", "B"]:
            if not os.path.isdir("%s/%s/test" % (self.data_path, site)):
                continue
            #site is the one to harmonize
            for subject_id in os.listdir(self.data_path + site + "/test/"):
                num_patches = np.load("%s/%s/test/%s/nb_patches.npy" % (self.data_path, site, subject_id))
                X = np.empty((num_patches, *self.input_shape, self.num_channels))
                
                # Generate data
                for i in range(num_patches):
                    #load patches
                    X[i] = np.load("%s/%s/test/%s/%s.npy" % (self.data_path, site, subject_id, i)).reshape((*self.input_shape, self.num_channels))
                y = self.generator_B.predict(X) if site == "A" else self.generator_A.predict(X)
                if not os.path.exists("%s/%s/infered/%s/" % (self.save_path, site, subject_id)):
                    os.makedirs("%s/%s/infered/%s/" % (self.save_path, site, subject_id))
                for i, pred in enumerate(y):
                    # pred + X[i] as we are trying to map the negative-residual
                    np.save("%s/%s/infered/%s/output_%s.npy" % (self.save_path, site, subject_id, i), pred)
        

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
                        help='Batch size required for training, by default 16')
    parser.add_argument("-lr", '--learning_rate', type = float, default = 0.0002,
                        help='Learning rate required to start our training, 10E-3 by default.')
    parser.add_argument("-s", '--strides', type = int, nargs = '+', default = 2,
                        help='Strides used for each convolutional layer, by default (2, 2, 2)')
    parser.add_argument("-ps", '--pool_size', type = int, nargs = '+', default = 2,
                        help='Size of each max pool layer, by default (2, 2, 2)')
    parser.add_argument("-fs", '--filter_size', type = int, nargs = '+', default = 4,
                        help='Filter size of our Unet model, by default (4,4,4)')
    parser.add_argument("-nc", '--num_channels', type = int, default = 1,
                        help='Number of contrasts as input, by default 1')
    parser.add_argument("-e", '--epochs', type = int, default = 10,
                        help='Number of training epochs , by default 10')
    parser.add_argument("-ff", '--features_factor', type = int, default = 16,
                        help='Factor by which multiply the powers of 2, will give the features number needed at each depth level , by default 16')
    parser.add_argument("-spe", '--steps_per_epoch', type = int, default = 25,
                        help='Number of steps for each epoch , by default 25')
    parser.add_argument("-wpd", '--weights_path_disc', type=str, default = None,
                        help = "path to the file containing already trained discriminator's weights. None by default")
    parser.add_argument("-wpga", '--weights_path_gen_a', type=str, default = None,
                        help = "path to the file containing already trained generator_A's weights. None by default")
    parser.add_argument("-wpgb", '--weights_path_gen_b', type=str, default = None,
                        help = "path to the file containing already trained generator_B's weights. None by default")
    parser.add_argument("-wp", '--weights_path', type=str, default = None,
                        help = "path to the file containing already trained model's weights. None by default")
    parser.add_argument("-sp", '--save_path', type=str, default = "./models/",
                        help = "path to the directory in which to save our models weights.")
    parser.add_argument('--infer_only', action = 'store_true',
                        help = "Indicates wether we want to train or not our model")
    parser.add_argument('--train_only', action = 'store_true',
                        help = "Indicates wether we want to infer or not our model")
    parser.add_argument('--batch_norm', action = 'store_true',
                        help = "Will add batchnormalization layers after each conv layer of encoder")
    parser.add_argument('--gpu', type = int, nargs = '+', default = [],
                        help='Which gpu are required, None by default')
    parser.add_argument('--optimizer', type = str, default = "adam",
                        help='Optimizer to use for training the model')
    parser.add_argument('-ra', "--ref_a", type = str, default = "A",
                        help='Data Reference A for the dualGAN')
    parser.add_argument('-rb', "--ref_b", type = str, default = "B",
                        help='Data Reference B for the dualGAN')
    parser.add_argument('--load_ckpt', action = 'store_true',
                        help = "Indicates wether we want to restore the last training checkpoint or not")
    parser.add_argument('--val_on_cpu', action = 'store_true',
                        help = "Indicates wether we want to run validation phase on CPU or not, can be useful as GPUs sometimes run out of memory in this phase")

    args = parser.parse_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if len(args.gpu) == 1 :
        os.environ["CUDA_VISIBLE_DEVICES"]="%s" % args.gpu[0]
    elif len(args.gpu) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"]="%s,%s" % tuple(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only use the first GPU
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         print(e)
    model = DualGAN(input_shape = tuple(args.input_shape), data_path = args.data_path,
                    num_channels = args.num_channels, batch_norm = args.batch_norm,
                    pool_size = args.pool_size, filter_shape = args.filter_size,
                    batch_size = args.batch_size, model_type = "dualGAN",
                    initial_learning_rate = args.learning_rate,
                    strides = args.strides, depth = args.depth, weights_path_disc = args.weights_path_disc,
                    weights_path_gen_A = args.weights_path_gen_a, weights_path_gen_B = args.weights_path_gen_b,
                    weights_path = args.weights_path,
                    save_path = args.save_path, features_factor = args.features_factor, optimizer = args.optimizer)
    if not args.infer_only:
        model.train(args.epochs, args.steps_per_epoch, args.load_ckpt, args.val_on_cpu)
        #model.model.load_weights(model.save_path + "/best.h5")
        model.model.save_weights(model.save_path + '/final_model_weights.h5')
        print("Models weights saved.")
    if not args.train_only:
        model.infer(args.load_ckpt)
        print("Inference done!")
    print("Exiting script")
