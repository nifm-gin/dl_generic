import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import ReLU, Add, Multiply, BatchNormalization, Flatten
from tensorflow.keras.activations import sigmoid
from augmentations import *
import os
import numpy as np
import sys
import time, datetime
import pickle
from ops import ModifiedReduceLROnPlateau, CustomExponentialDecay, Scheduler, contrastive_loss

class Model3D(object):

    def __init__(self, input_shape, data_path , num_channels = 1, model = None, pool_size=(2, 2, 2),
                 filter_shape=(3, 3, 3), dropout=.1, batch_norm=False, initial_learning_rate=0.00001, batch_size = 8,
                 model_type='regression', padding='same', strides = (1, 1, 1), weights_path = None, activation = "relu",
                 save_path = "models/", name = "Model", decay_rate = None, data_augmentation = []):

        """
        Inspired of DeepRad source code : "https://github.com/QTIM-Lab/DeepRad/blob/master/deepneuro/models/model.py"
        Mother class of future model, contains universal parameters and methods
        Model.model will be a Keras model instance
        By default is in 3d but can be heritated by 2D models as long as specifications hold
        
        Parameters
        ----------
        input_shape : tuple, optional
            Input dimensions of first layer. Not counting batch-size.
        input_tensor : tensor, optional, TODO later 
            If input_tensor is specified, build_model will output a tensor
            created from input_tensor.
        num_channels : int, optional
            specifies the number of contrasts present in the input
        pool_size : tuple, optional
            Pool size for convolutional layers.
        filter_shape : tuple, optional
            Filter size for convolutional layers.
        dropout : float, optional
            Dropout percentage for children models. Each model's implementation of
            dropout will differ.
        batch_norm : bool, optional
            Whether layers are batch-normed in children models. Each model's implementation
            of which layers will be batch-normed is different.
        initial_learning_rate : float, optional
            Initial learning rate for the chosen optimizer type, if necessary
        batch_size : int, optional
            Batch size wanted for training
        model_type : str, optional
            indicate the type of our model : "segmentation" "regression" "generation" ...
        activation : str, optional
            What type of activation to use at each layer. May be implemented differently in
            each model.
        padding : str, optional
            Padding for convolutional layers.
        strides : tuple, optional
            specifies the stride at each convolutional layer
        weights_path : str, optional
            path to the model weights to load if a model has been pre-trained 
        data_path : str, optional
            path to the generated patches, organized as path/train(val)/patches/...
        save_path : str, optional
            path where to save the model and the infered data that went through the model
        """
        #Generic Model Parameters -- Optional
        self.input_shape = input_shape
        self.num_channels = num_channels
        #Hyper parameters
        self.pool_size = pool_size
        self.filter_shape = filter_shape
        self.batch_size = batch_size
        self.filter_shape = filter_shape
        self.strides = strides
        self.decay_rate = decay_rate
        if decay_rate is None:
            self.learning_rate = initial_learning_rate
        else:
            self.learning_rate = CustomExponentialDecay(
                initial_learning_rate,
                decay_steps=10000,
                decay_rate=decay_rate,
                staircase=True
            )
        
        self.name = name
        #architecture params
        self.padding = padding
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.data_path = data_path
        self.model_type = model_type
        self.save_path = save_path
        #learning param
        self.weights_init = RandomNormal(stddev = 0.05)
        
        self.optimizer = [Adam(self.learning_rate, beta_1=0.5)]
        self.class_weights = None
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.writer = tf.summary.create_file_writer(self.save_path + "/logs/fit/" + self.name + datetime.datetime.now().strftime("%d-%H%M"))
        self.scheduler = Scheduler(self.optimizer, [self.name], early_stopping = 30)
        self.augmentations = [dict_augmentations[name] for name in data_augmentation]
        # Derived Parameters
        self.model = model
        self.features_extractor = None
        if self.model is None:
            self.build_model()
        #self.model.summary()
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if weights_path is not None:
            try:
                assert os.path.isfile(weights_path), "Weights path specified does not exist, models kept as if"
                self.model.load_weights(weights_path)
                print("Model's weights loaded")
            except AssertionError as err:
               print(err)
            except Exception as err:
                print("Error raised while loading weights : %s.\n\tContinuing with model as if" % err)
        
    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """
        pass
    
    def build_model(self):

        """ This method is inherited by child classes to specify the classes model attribute. If input_tensor
            is specified, build_model returns a tensor output. If not, it return a Keras model output.
        """
        pass

    def restore_ckpt(self):
        checkpoint_path = self.save_path + "/checkpoints/train"
        ckpt = tf.train.Checkpoint(model=self.model,
                                   model_optimizer=self.optimizer[0])
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        try:
            if ckpt_manager.latest_checkpoint:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                print ('Latest checkpoint restored!!')
            else:
                print("Could not find any saved checkpoints")
        except Exception as err:
            print("Could not restore latest checkpoint, continuing as if!", err)

    #@tf.function
    def train_step(self, inp, y):
        #Shifting input and target doesnt make sense here as our images always have the same dimensions
        with tf.GradientTape() as tape:
            predictions = self.model(inp, training = True)
            #tf.print(y, predictions)
            if self.class_weights != None:
                if self.num_classes == 2:
                    sample_weight = np.ones_like(y)
                    for i in range(self.num_classes):
                        sample_weight[y == i] = self.class_weights[i]
                else:
                    sample_weight = np.dot(y, self.class_weights)
                loss = self.loss(y, predictions, sample_weight = sample_weight)
            else:
                loss = self.loss(y, predictions)
        self.train_loss(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)    
        self.optimizer[0].apply_gradients(zip(gradients, self.model.trainable_variables))
        return {'training_loss' : loss}

#    @tf.function
    def val_step(self, inp, tar, on_cpu):
        with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
            predictions = self.model(inp, training = False)
            if self.class_weights != None:
                if self.num_classes == 2:
                    sample_weight = np.ones_like(tar)
                    for i in range(self.num_classes):
                        sample_weight[tar == i] = self.class_weights[i]
                else:
                    sample_weight = np.dot(tar, self.class_weights)
                loss = self.loss(tar, predictions, sample_weight = sample_weight)
            else:
                loss = self.loss(tar, predictions)
            return {'val_loss' : loss}
        
    def burn_steps(self, n_steps):
        print("Burning %s steps" % n_steps)
        for cpt, (X, y) in enumerate(self.data_generator()):
            if cpt == n_steps or X is None:
                return
            self.train_step(X, y)
        #Restore true optimizer so we can start training properly

    def get_checkpoint(self):
        return tf.train.Checkpoint(model=self.model,
                                   model_optimizer=self.optimizer[0])
    
    def train(self, epochs = 10, val_on_cpu = False):
        """
        method to train our model
        """ 
        checkpoint_path = self.save_path + "/checkpoints/train"
        ckpt = self.get_checkpoint()
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        print("Start training")
        #loading validation data only once
        with self.writer.as_default():
            for e in range(1, epochs):
                metrics, cpt = None, 0
                start = time.time()
                self.train_loss.reset_states()
                for (X, y) in self.data_generator():
                    if X is None:
                        break
                    if metrics is None:
                        metrics  = self.train_step(X, y)
                    else:
                        c_metrics = self.train_step(X, y)
                        for key in c_metrics:
                            metrics[key] += c_metrics[key]
                    cpt+=1
                print('Loss for Epoch %s : %s' % (e, self.train_loss.result().numpy()))
                for key in metrics :
                    tf.summary.scalar(key, metrics[key] / cpt, step = e)
                try :
                    tf.summary.scalar("LR", self.optimizer_gen.lr.lr, step = e)
                except:
                    tf.summary.scalar("LR", self.optimizer_gen.lr, step = e)

                #Validation step
                if e % 5 == 0:
                    metrics, cpt = None, 0
                    for (X, y) in self.generator_val():
                        if X is None:
                            break
                        if metrics is None:
                            metrics = self.val_step(X, y)
                        else:
                            c_metrics = self.val_step(X, y)
                            for key in c_metrics:
                                metrics[key] += c_metrics[key]
                        cpt+=1
                    val_loss = self.get_scheduler_losses(metrics)
                    print("Val loss : %s " %  val_loss[0])
                    for key in metrics :
                        tf.summary.scalar(key, metrics[key] / cpt, step = e)
                    scheduler_output = self.scheduler.update(val_loss)
                    if scheduler_output == "stop":
                        print('Early stopping reached !')
                        break
                    elif scheduler_output:
                        ckpt_save_path = ckpt_manager.save()
                        print ('Saving checkpoint for epoch {} at {}'.format(e,
                                                                         ckpt_save_path))
                print ('Time taken for epoch {} is {} sec\n'.format(e,
                                                                time.time()-start))
        #Restore best model & Save model
        self.writer.flush()
        print("Restoring  best checkpoint")
        ckpt.restore(ckpt_manager.latest_checkpoint)
        self.model.save_weights(self.save_path + '/best.h5')

    def  get_scheduler_losses(self, metrics):
        return [metrics[key] for key in metrics]
    
    def self_pretrain(self, epochs = 10, steps = 25, val_on_cpu = False):
        """
        Implementation of self pretraining  using contrastive loss
        """
        assert self.features_extractor != None, "No features_extractor model defined."
        checkpoint_path = self.save_path + "/checkpoints/train"
        ckpt = self.get_checkpoint()
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        print("Start self pre-training")
        pretraining_scheduler = Scheduler([self.optimizer[0]], [self.name], early_stopping = 30)
        #loading validation data only once
        X1_val, X2_val, Y_val = self.pretraining_val_generator()
        for e in range(1, epochs):
            for cpt, (X1, X2, y) in enumerate(self.pretraining_generator()):
                self.self_pretrain_step(X1, X2, y, e)
                if cpt == steps:
                    break

            if e % 10 == 0:
                val_loss = self.self_pretrain_val_step(X1_val, X2_val, Y_val, e, val_on_cpu)
                print("Self pre-training Val loss : %s " %  (val_loss[0].numpy() if isinstance(val_loss, list) else val_loss.numpy()))
                scheduler_output = pretraining_scheduler.update(val_loss if isinstance(val_loss, list) else [val_loss])
                if scheduler_output == "stop":
                    print('Early stopping reached !')
                    break
                elif scheduler_output:
                    ckpt_save_path = ckpt_manager.save()
                    print ('Saving pretraining checkpoint for epoch {} at {}'.format(e,
                                                                         ckpt_save_path))
                
        try:
            self.optimizer[0].lr.assign(self.learning_rate)
        except:
            self.optimizer[0].lr.lr = self.learning_rate

        #Restore best pretrained model
        ckpt.restore(ckpt_manager.latest_checkpoint)

    def self_pretrain_step(self, X1, X2, y, epoch):
        with tf.GradientTape(persistent=True) as tape:
            z1 = self.features_extractor(X1, training=True)
            z2 = self.features_extractor(X2, training=True)
            loss = contrastive_loss(z1, z2, y)
        gradients = tape.gradient(loss, 
                                  self.features_extractor.trainable_variables)
        self.optimizer[0].apply_gradients(zip(gradients, 
                                               self.features_extractor.trainable_variables))
        epoch = tf.constant(epoch, dtype = tf.int64)
        with self.writer.as_default():
            tf.summary.scalar('contrastive_loss', loss, step=epoch)

    def self_pretrain_val_step(self, X1, X2, y, epoch, on_cpu = False):
        with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
            z1 = self.features_extractor(X1, training=False)
            z2 = self.features_extractor(X2, training=False)
            loss = contrastive_loss(Flatten()(z1), Flatten()(z2), y)
            epoch = tf.constant(epoch, dtype = tf.int64)
            with self.writer.as_default():
                tf.summary.scalar('contrastive_val_loss', loss, step=epoch)
            return loss

    def pretraining_generator(self):
        """
        iterator that feeds our pre-training
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/train/nb_patches.npy" % (self.data_path))
        X1 = np.empty((self.batch_size, *self.input_shape, self.num_channels), dtype = np.float32)
        X2 = np.empty((self.batch_size, *self.input_shape, self.num_channels), dtype = np.float32)
        subjects1 = np.empty((self.batch_size,))
        subjects2 = np.empty((self.batch_size,))
        sub_dict = None
        with open("%s/train/subjects_patches.pkl" % (self.data_path), "rb") as f:
            sub_dict = pickle.load(f)
        while True:
            indexes = np.random.choice(num_patches, self.batch_size)
            for i, ID in enumerate(indexes):
                data = np.load("%s/train/%s.npz" % (self.data_path, ID))
                X1[i] = data["data"].reshape((*self.input_shape, self.num_channels))
                subjects1[i] = data["sub"]
                # we want as many  positive pairs than  negative ones
                if np.random.uniform() < 0.5:
                    #we want a patch from same subject
                    ID2 = np.random.randint(sub_dict[subjects1[i]][0], sub_dict[subjects1[i]][1])
                else:
                    try:
                        ID2 = np.random.choice(np.arange(0, sub_dict[subjects1[i]][0]) + np.arange(sub_dict[subjects1[i]][1], num_patches))
                    except:
                        if sub_dict[subjects1[i]][1] == num_patches:
                            ID2 = np.random.choice(np.arange(0, sub_dict[subjects1[i]][0]))
                        else:
                            ID2 = np.random.choice(np.arange(sub_dict[subjects1[i]][1], num_patches))
                data2 = np.load("%s/train/%s.npz" % (self.data_path, ID2))
                X2[i] = data2["data"].reshape((*self.input_shape, self.num_channels))
                subjects2[i] = data2["sub"]
            yield X1, X2, subjects1 == subjects2

    def pretraining_val_generator(self):
        """
        iterator that feeds our pre-training
        ran in parrallel to training (meant to)
        """
         
        num_patches = np.load("%s/val/nb_patches.npy" % (self.data_path))
        X1 = np.empty((num_patches, *self.input_shape, self.num_channels), dtype = np.float32)
        X2 = np.empty((num_patches, *self.input_shape, self.num_channels), dtype = np.float32)
        subjects1 = np.empty((num_patches,))
        subjects2 = np.empty((num_patches,))
        sub_dict = None
        with open("%s/val/subjects_patches.pkl" % (self.data_path), "rb") as f:
            sub_dict = pickle.load(f)
        for i in range(num_patches):
            data = np.load("%s/val/%s.npz" % (self.data_path, i))
            X1[i] = data["data"].reshape((*self.input_shape, self.num_channels))
            subjects1[i] = data["sub"]
            ID2 = np.random.choice(num_patches)
            # we want as many  positive pairs than  negative ones
            if np.random.uniform() < 0.5:
                #we want a patch from same subject
                ID2 = np.random.randint(sub_dict[subjects1[i]][0], sub_dict[subjects1[i]][1])
            else:
                try:
                    ID2 = np.random.choice(np.arange(0, sub_dict[subjects1[i]][0]) + np.arange(sub_dict[subjects1[i]][1], num_patches))
                except:
                    if sub_dict[subjects1[i]][1] == num_patches:
                        ID2 = np.random.choice(np.arange(0, sub_dict[subjects1[i]][0]))
                    else:
                        ID2 = np.random.choice(np.arange(sub_dict[subjects1[i]][1], num_patches))
            data2 = np.load("%s/val/%s.npz" % (self.data_path, ID2))
            X2[i] = data2["data"].reshape((*self.input_shape, self.num_channels))
            subjects2[i] = data2["sub"]
        return X1, X2, subjects1 == subjects2
            
    def generator_val(self):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/val/nb_patches.npy" % (self.data_path))
        num_steps = num_patches // self.batch_size
        X = np.empty((num_patches, *self.input_shape, self.num_channels))
        y = np.empty((num_patches, *self.input_shape, 1))
        for i in range(num_steps):
            indexes = np.arange(i * self.batch_size, (i + 1) * self.batch_size)
            # Generate data
            for i, ID in enumerate(indexes):
                X[i] = np.load("%s/val/%s.npz" % (self.data_path, ID))["data"].reshape((*self.input_shape, self.num_channels))
                y[i] = np.load("%s/val/output_%s.npy" % (self.data_path, ID)).reshape((*self.input_shape, 1))
                yield X, y
        yield None, None
        
    def data_generator(self, data_type = 'train'):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/%s/nb_patches.npy" % (self.data_path, data_type))
        num_steps = num_patches // self.batch_size
        indices = np.arange(num_patches); np.random.shuffle(indices)
        X = np.empty((self.batch_size, *self.input_shape, self.num_channels))
        y = np.empty((self.batch_size, *self.input_shape, 1))
        for i in range(num_steps):
            indexes = indices[i * self.batch_size : (i + 1) * self.batch_size]
            # Generate data
            for i, ID in enumerate(indexes):
                #load patches
                X[i] = np.load("%s/%s/%s.npz" % (self.data_path, data_type, ID))["data"].reshape((*self.input_shape, self.num_channels))
                y[i] = np.load("%s/%s/output_%s.npy" % (self.data_path, data_type, ID)).reshape((*self.input_shape, 1))
                yield X, y.reshape((indexes.shape[0], *self.input_shape, 1))
        yield None, None

    def infer(self, is_gan = False, saliency = False):
        """
        method infering test data with our model
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
                X[i] = np.load("%s/test/%s/%s.npz" % (self.data_path, subject_id, i))["data"].reshape((*self.input_shape, self.num_channels))
            if is_gan:
                y = self.generator.predict(X)
            else:
                y = self.model.predict(X)
            if not os.path.exists("%s/infered/%s/" % (self.save_path, subject_id)):
                os.makedirs("%s/infered/%s/" % (self.save_path, subject_id))
            for i, pred in enumerate(y):
                np.save("%s/infered/%s/output_%s.npy" % (self.save_path, subject_id, i), pred)
            if saliency:
                saliency = self.saliency(X)
                for i, patch in enumerate(saliency):
                    np.save("%s/infered/%s/saliency_%s.npy" % (self.save_path, subject_id, i), patch)
            
    def saliency(self, x):
        """
        Method computing saliency map for an input x (n_patches, x, y (,z), n_channels)
        """
        print("Processing Saliency map")
        image = tf.Variable(x if self.additional_inputs is None else x[0], dtype=float)
        with tf.GradientTape() as tape:
            pred = self.model(image if self.additional_inputs is None else [image, x[1]], training=False)
            if self.num_classes == 2:
                #linear activation, we will use it to feed our saliency workflow (from_logit = True)
                loss =  pred
            else:
                class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
                loss = pred[0][class_idxs_sorted[0][0]]
        x = x[0] if self.additional_inputs is not None else x
        grads = tape.gradient(loss, image)
        dgrad_abs = tf.math.abs(grads)
        dgrad_max_ = np.max(dgrad_abs, axis = -1)
        if len(self.input_shape) == 2:
            arr_min, arr_max  = np.min(dgrad_max_, axis = (1,2)).reshape((x.shape[0],1,1)), np.max(dgrad_max_, axis = (1,2)).reshape((x.shape[0],1,1))
        else:
            arr_min, arr_max  = np.min(dgrad_max_, axis = (1,2,3)).reshape((x.shape[0],1,1,1)), np.max(dgrad_max_, axis = (1,2,3)).reshape((x.shape[0],1,1,1))
        saliency = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
        return saliency.reshape(x.shape[:-1] + (1,))
