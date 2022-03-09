import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, ReLU
import os
import numpy as np
from tensorflow.keras import Model, Input
import sys
from VAEGAN import VAEGAN 
import time
from Models import Model3D
import pickle
from augmentations import *
from tensorflow.keras.regularizers import l1
from ops import categorical_ce, binary_ce, loss_obj, discriminator_loss, Scheduler, vae_sigmoid_loss_logits, content_loss, ssim_loss, style_loss, contrastive_loss
####################################################################################################

# OT-DA GAN model #

####################################################################################################
class ImUnity(VAEGAN):
    """
    ImUnity model that heritates from VAEGAN class
    Includes additional modules (bio & site) related to harmonization purposes
    """
    def __init__(self, input_shape, data_path, num_channels = 1, model = None, pool_size=2, filter_shape= 3, dropout=.1, batch_norm=False,
                 initial_learning_rate=0.001, batch_size = 8, model_type = "harmonization", padding='same', strides = 1, weights_path = None, class_weights = None,
                 save_path = None, activation = "relu", depth = 4, features_factor = 1, name = "ImUnity", num_sites = 2, decay_rate = None, latent_dim = 2, data_augmentation = [],
                 loss_ratios = [1, 1], num_sequences = 1, sequence_weights = None, biological_features=None):
        """
        parameters:
        """
        #check dimension of model :
        self.num_sites = num_sites
        self.num_sequences = num_sequences
        self.biological_features = biological_features 
        super(ImUnity, self).__init__(input_shape, data_path, num_channels, model, pool_size, filter_shape, dropout, batch_norm, initial_learning_rate, batch_size, model_type, padding, strides, weights_path, activation, depth = depth, save_path = save_path, name = name, decay_rate = decay_rate, features_factor = features_factor, latent_dim = latent_dim, contrast_encoder = True)
        if class_weights is not None:
            assert len(class_weights) == self.num_sites, 'Error with class_weights for sites_module loss'
        self.class_weights = class_weights
        self.sequence_weights = sequence_weights

        #setting up modules
        self.modules_loss = [binary_ce if self.num_sites == 2 else categorical_ce]
        self.loss_ratios = loss_ratios
        if biological_features is not None:
            self.modules_loss += [binary_ce for f in biological_features]
            assert len(self.loss_ratios) == len(self.biological_features) + 2, "Error in Loss ratios, it must match number  of modules (Group + biological ones + Sequence)" 
        self.modules_loss += [binary_ce if self.num_sequences == 2 else categorical_ce] * (self.num_sequences > 1)  
        print(self.modules_loss)
        self.optimizer_modules = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.optimizer.append(self.optimizer_modules)
        self.scheduler = Scheduler(self.optimizer,
                                   names = ["Gen", "Disc", "Modules"],
                                   ratios = [1, 1, 1],
                                   patience = 3, early_stopping = 30)
        
    def build_model(self):
        #init
        super(ImUnity, self).build_model()
        gan_inputs = [Input(self.input_shape + (self.num_channels,)), Input(self.input_shape + (self.num_channels,))]
        generated_image, z, mean, logvar, z_anat = self.generator(gan_inputs)
        
        #Building discriminator :
        discriminator_output = self.discriminator(generated_image)

        #Modules densenet taking as input z latent vector
        self.z_shape = z_anat.shape[1:]
        modules_input = Input(self.z_shape)
        f = Flatten()(modules_input)
        self.weights_init = tf.keras.initializers.RandomNormal(stddev = 0.5)
        modules_outputs = []
        f_site = Dense(128, kernel_initializer = self.weights_init, activity_regularizer=l1(0.0001))(f)
        f_site = ReLU()(f_site)
        site = Dense(1 if self.num_sites == 2 else self.num_sites, kernel_initializer = self.weights_init)(f_site)
        if self.biological_features is not None:
            for _ in self.biological_features:
                f_d = Dense(128, kernel_initializer = self.weights_init, activity_regularizer=l1(0.0001))(f)
                f_d  = ReLU()(f_d)
                modules_outputs.append(Dense(1, kernel_initializer = self.weights_init)(f_d))
        if self.num_sequences > 1:
            #we try to harmonize multiple sequence types at once here
            f_sequence = Dense(128, kernel_initializer = self.weights_init, activity_regularizer=l1(0.0001))(f)
            sequence = Dense(1 if self.num_sequences == 2 else self.num_sequences, kernel_initializer = self.weights_init)(f_sequence)
            self.modules = Model(inputs = modules_input, outputs = [site,] + modules_outputs + [sequences], name = "Modules")
        else:
            self.modules = Model(inputs = modules_input, outputs = [site,] + modules_outputs, name = "Modules")
        
        modules_outputs = self.modules(z_anat)
        #Update self instance with built model
        self.model = Model(inputs = gan_inputs,
                           outputs = [generated_image, discriminator_output, modules_outputs],
                           name = self.name)
        self.modules.summary()
        self.model.summary()
        
    def get_checkpoint(self):
        return tf.train.Checkpoint(generator=self.generator,
                                   discriminator = self.discriminator,
                                   modules = self.modules,
                                   model = self.model
        )
    def restore_ckpt(self):
        checkpoint_path = self.save_path + "/checkpoints/train"
        ckpt = self.get_checkpoint()
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

    def dagan_generator_loss(self, x, generated, x_gen, style, z_style, z, mean, logvar, z_generated, disc_generated_output, modules_generated_output, target):
        z, z_generated, z_style = Flatten()(z), Flatten()(z_generated), Flatten()(z_style)
        mask = tf.math.greater_equal(x, tf.constant(0.001, dtype = tf.float32))

        #we want site_module_prediction distribution to be as close to uniform as possible
        target[0] = np.ones_like(target[0], dtype = np.float32) / self.num_sites
        generated = tf.where(mask, tf.sigmoid(generated), x)
        gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
        gan_loss_modules = sum([self.loss_ratios[i] * loss(target[i], modules_generated_output[i]) for i, loss in enumerate(self.modules_loss)]) / np.sum(self.loss_ratios)
        #L1 coss measures mae between latent space inputs representation
        l1_loss_cost = tf.reduce_mean(tf.abs(z - z_generated))
        if l1_loss_cost == 0:
            #in case extracted features are always the same
            l1_loss_cost = 1
        #l1 shift measures the variation induced by the model
        l1_loss_shift = tf.reduce_mean(tf.boolean_mask(tf.abs(x - x_gen), mask))
        #l1 style measures mae between harmonized input and GT
        l1_loss_style = tf.reduce_mean(tf.boolean_mask(tf.abs(style - generated), mask))

        #kl loss measure latent inputs space distribution distance to gaussian distribution
        kl = 0.5 * tf.math.reduce_mean(tf.math.exp(logvar) + tf.math.square(mean) - 1. - logvar)

        #ssim loss getseen harmonized input and GT
        ss_loss = ssim_loss(generated, style)
        
        #Total loss composed of all losses presented above
        total_gen_loss = 2E0 * gan_loss + 1E2 * l1_loss_style + 1E-2 * gan_loss_modules + 1E-6 * kl + 1E0 * ss_loss# + 0E0 * l1_loss_shift
        return total_gen_loss, gan_loss, gan_loss_modules, l1_loss_cost, l1_loss_style, kl, l1_loss_shift, ss_loss


    def train_step(self, X, y):
        Y = y[0][0]
        X_gamma = y[0][1]
        y = y[1:]
        sample_weight = np.empty(self.batch_size)
        style_output = Y
        with tf.GradientTape(persistent=True) as tape:
            (gen_output, z, mean, logvar, z_anat) = self.generator([X, Y], training=True)
            (_, z_style, _, _, _) = self.generator([Y, Y], training=True)
            #uncomment line if you want to force indentity preservation for contrast=anatomical  input
            #(X_gen, _, _, _, _) = self.generator([X, X], training=True)
            X_gen = X.copy()
            #We could also consider latent space representation of input and harmonized data (should be close in the anatomical latent space) and force this closeness using contrastive loss
            gen_output_sig = tf.sigmoid(gen_output)
            (_, z_generated, _, _, _) = self.generator([gen_output_sig, Y], training=True)
            #We will train our disc to determine between style site and others
            disc_real_output = self.discriminator(X_gamma, training=True)
            disc_fake_output = self.discriminator(gen_output_sig, training=True)
            modules_output = self.modules(z_anat, training = True)
            if self.biological_features is None and self.num_sequences == 1:
                modules_output = [modules_output]
            # compute losses
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
            total_gen_loss, gan_loss, gen_loss_modules, l1_loss_cost, l1_loss_style, kl, l1_loss_shift, ss_loss = self.dagan_generator_loss(X, gen_output, X_gen, X_gamma, z_style, z, mean, logvar, z_generated, disc_fake_output, modules_output, y.copy())
            #We create the associate sample_weight for computing the loss (sites might not be equally represented)
            sample_weight = [np.dot(y[0], self.class_weights) if self.class_weights is not None else None, None, None]
            if self.num_sequences > 1:
                #we do the same for sequences
                sample_weight.append([self.sequence_weights[int(i[0])] for i in y[-1]] if self.num_sequences == 2 else np.dot(y[-1], self.sequence_weights))
            modules_loss = sum([self.loss_ratios[i] * loss(y[i], modules_output[i], sample_weight = sample_weight[i]) for i, loss in enumerate(self.modules_loss)])

        # Compute gradients for generator and discriminator
        self.train_loss(total_gen_loss)
        generator_gradients = tape.gradient(total_gen_loss, 
                                            self.generator.trainable_variables)
        discriminator_gradients = tape.gradient(disc_loss, 
                                                self.discriminator.trainable_variables)
        modules_gradients = tape.gradient(modules_loss, 
                                          self.modules.trainable_variables)
        
        # Apply gradients to optimizers
        self.optimizer_gen.apply_gradients(zip(generator_gradients, 
                                               self.generator.trainable_variables))
        self.optimizer_disc.apply_gradients(zip(discriminator_gradients,
                                                self.discriminator.trainable_variables))
        self.optimizer_modules.apply_gradients(zip(modules_gradients,
                                                   self.modules.trainable_variables))
        return {'total_gen_loss': total_gen_loss,
                'l1_loss_cost' : l1_loss_cost,
                'l1_loss_shift' : l1_loss_shift,
                'l1_loss_style' : l1_loss_style,
                'disc_loss' : disc_loss,
                'gan_disc_loss' : gan_loss,
                'modules_loss' : modules_loss,
                'modules_confusion_loss' : gen_loss_modules,
                'kl_loss' : kl,
                'ssim_loss' : ss_loss}

    def val_step(self, X, y, on_cpu = False):
        Y = y[0][0]
        X_gamma = y[0][1]
        y = y[1:]
        style_output = Y
        with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
            (gen_output, z, mean, logvar, z_anat) = self.generator([X, Y], training=False)
            (_, z_style, _, _, _) = self.generator([Y, Y], training=False)
            #(X_gen, _, _, _, _) = self.generator([X, X], training=False)
            X_gen = X.copy()
            gen_output_sig = tf.sigmoid(gen_output)
            (_, z_generated, _, _, _) = self.generator([gen_output_sig, Y], training=False)
            disc_real_output = self.discriminator(X_gamma, training=False)
            disc_fake_output = self.discriminator(gen_output_sig, training=False)
            modules_output = self.modules(z_anat, training = False)
            if self.biological_features is None and self.num_sequences == 1:
                modules_output = [modules_output]
            # compute losses
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
            total_gen_loss, gan_loss, gen_loss_modules, l1_loss_cost, l1_loss_style, kl, l1_loss_shift, ss_loss = self.dagan_generator_loss(X, gen_output, X_gen, X_gamma, z_style, z, mean, logvar, z_generated, disc_fake_output, modules_output, y.copy())
            modules_loss = sum([self.loss_ratios[i] * loss(y[i], modules_output[i]) for i, loss in enumerate(self.modules_loss)])
            return {'total_gen_val_loss' : total_gen_loss,
                'l1_val_loss_cost' : l1_loss_cost,
                'l1_loss_shift' : l1_loss_shift,
                'l1_loss_style_val' : l1_loss_style,
                'disc_val_loss' : disc_loss,
                'gan_disc_val_loss' : gan_loss,
                'modules_val_loss' : modules_loss,
                'modules_confusion_val_loss' : gen_loss_modules,
                'kl_val_loss' : kl,
                'ssim_val_loss' : ss_loss}

    def  get_scheduler_losses(self, metrics):
        return [metrics[key] for key in ['total_gen_val_loss', 'disc_val_loss', 'modules_val_loss']]
    
    
    def data_generator(self):
        num_patches = np.load("%s/train/nb_patches.npy" % (self.data_path))
        num_steps = num_patches // self.batch_size
        X = np.empty((self.batch_size, *self.input_shape, self.num_channels), dtype = np.float32)
        Y = np.empty((2, self.batch_size, *self.input_shape, self.num_channels), dtype = np.float32)
        sub_dict = None
        indices = np.arange(num_patches); np.random.shuffle(indices)
        with open("%s/train/subjects_patches.pkl" % (self.data_path), "rb") as f:
            sub_dict = pickle.load(f)
        for i in range(num_steps):
            label_site = np.zeros((self.batch_size, 1 if self.num_sites == 2 else self.num_sites))
            label_sequence = np.zeros((self.batch_size, 1 if self.num_sequences == 2 else self.num_sequences))
            labels_bio = np.zeros((len(self.biological_features), self.batch_size,1)) if self.biological_features is not None else []
            indexes = indices[i * self.batch_size : (i + 1) * self.batch_size]
            # Generate data
            for i, ID in enumerate(indexes):
                #load patches
                data = np.load("%s/train/%s.npz" % (self.data_path, ID), allow_pickle = True)
                X[i] = data['data'].reshape((*self.input_shape, self.num_channels))
                sub1 = float(data["sub"])
                ID2 = np.random.randint(sub_dict[sub1][0], sub_dict[sub1][1])
                data2 = np.load("%s/train/%s.npz" % (self.data_path, ID2))
                #In Y : 0) another  slice from same subject (same contrast but different structure)
                # 1) the same slice
                #We  then will compute a gamma function on these slices
                Y[0, i] = data2['data'].reshape((*self.input_shape, self.num_channels))
                Y[1, i] = X[i] 
                Y[:, i] = gamma(Y[:, i])
                if self.num_sites == 2:
                    label_site[i] = data["group"].item()["Group"]
                else:
                    label_site[i][int(data["group"].item()["Group"])] = 1
                bio_f = []
                try:
                    if self.biological_features	is not None :
                        for n, name in enumerate(self.biological_features):
                            if self.loss_ratios[n] > 0:
                                labels_bio[n, i] = data["group"].item()[name]
                    if self.num_sequences == 2:
                        label_sequence[i] = data["group"].item()["Sequence"]
                    elif self.num_sequences > 2:
                        label_sequence[i][int(data["group"].item()["Sequence"])] = 1
                except Exception as err:
                    print(err)
                
            for augmentation in self.augmentations:
                for i in range(self.batch_size):
                    if np.random.uniform() < (0.5 / len(self.augmentations)):
                        X[i] = augmentation(X[i])
            labels_bio = [l for l in labels_bio]
            if self.biological_features is not None:
                yield X, [Y, label_site] + labels_bio + [label_sequence] * (self.num_sequences > 1)
            else:
                yield X, [Y, label_site] + [label_sequence] * (self.num_sequences > 1)
        yield None, None
                
    def generator_val(self):
        """
        iterator that feeds our model
        ran in parrallel to training (meant to)
        """
        num_patches = np.load("%s/val/nb_patches.npy" % (self.data_path))
        num_steps = num_patches // self.batch_size
        X = np.empty((self.batch_size, *self.input_shape, self.num_channels), dtype = np.float32)
        Y = np.empty((2, self.batch_size, *self.input_shape, self.num_channels), dtype = np.float32)
        y_site = np.zeros((self.batch_size, 1 if self.num_sites == 2 else self.num_sites))
        labels_bio = np.zeros((len(self.biological_features), self.batch_size, 1)) if self.biological_features is not None else []
        y_sequence = np.zeros((self.batch_size, 1 if self.num_sequences == 2 else self.num_sequences))
        sub_dict = None
        with open("%s/val/subjects_patches.pkl" % (self.data_path), "rb") as f:
            sub_dict = pickle.load(f)
        for i in range(num_steps):
            indexes = np.arange(i * self.batch_size, (i + 1) * self.batch_size)
            # Generate data
            for i, ID in enumerate(indexes):
                data = np.load("%s/val/%s.npz" % (self.data_path, ID), allow_pickle = True)
                X[i] = data["data"].reshape((*self.input_shape, self.num_channels))
                sub1 = float(data["sub"])
                ID2 = np.random.randint(sub_dict[sub1][0], sub_dict[sub1][1])
                data2 = np.load("%s/val/%s.npz" % (self.data_path, ID2))
                Y[0, i] = data2["data"].reshape((*self.input_shape, self.num_channels))
                Y[1, i] = X[i] 
                Y[:, i] = gamma(Y[:, i])
                if self.num_sites == 2:
                    y_site[i] = data["group"].item()["Group"]
                else:
                    y_site[i][int(data["group"].item()["Group"])] = 1
                try:
                    if self.biological_features is not None :
                        for n, name in enumerate(self.biological_features):
                            if self.loss_ratios[n] > 0:
                                labels_bio[n, i] = data["group"].item()[name]
                    if self.num_sequences == 2:
                        y_sequence[i] = data["group"].item()["Sequence"]
                    elif self.num_sequences > 2:
                        y_sequence[i][int(data["group"].item()["Sequence"])] = 1
                except:
                    #case no sex or asd specified
                    pass
            labels_bio = [l for l in labels_bio]
            if self.biological_features is not None:
                yield X, [Y, y_site] + labels_bio + [y_sequence] * (self.num_sequences > 1)
            else:
                yield X, [Y, y_site] + [y_sequence] * (self.num_sequences > 1)
        yield None, None
        
    def infer_subject(self, sub_path, contrast, num_patches):
        X = np.empty((num_patches, *self.input_shape, self.num_channels))
        y = np.empty((num_patches, *self.input_shape, 1))
        z = np.empty((num_patches, *self.z_shape))
        # Generate data
        for i in range(num_patches):
            #load patches
            X[i] = np.load("%s/%s.npz" % (sub_path, i))["data"].reshape((*self.input_shape, self.num_channels))
        y, z  = self.generator.predict([X, contrast])[0:2]
        y = tf.sigmoid(y)
        y = np.where(X > 0.001, y, X)
        return y, z
    
    def infer(self, saliency = False):
        cpt, num_patches = 0, None
        for subject_id in os.listdir(self.data_path + "test/"):
            num_patches = np.load("%s/test/%s/nb_patches.npy" % (self.data_path, subject_id))
            break
        
        #classical inference script through MR and CT data
        #mr : index 0 / ct index : 1
        
        contrast_loaded = [False for i in range(self.num_sequences)]
        contrast = np.empty((self.num_sequences, num_patches, *self.input_shape, self.num_channels))
        for subject_id in os.listdir(self.data_path + "test/"):
            index = int("CT" in subject_id)
            if not contrast_loaded[index]:
                print("Taking subject : %s as contrast reference for %s scans" % (subject_id, ["mr", "ct"][index]))
                contrast_slice = np.load("%s/test/%s/%s.npz" % (self.data_path, subject_id, num_patches // 2))["data"].reshape((*self.input_shape, self.num_channels))
                for i in range(num_patches):
                    contrast[index, i] = contrast_slice
                contrast_loaded[index] = True            
            y, z = self.infer_subject("%s/test/%s/" % (self.data_path, subject_id), contrast[index], num_patches)
            if not os.path.exists("%s/infered/%s/" % (self.save_path, subject_id)):
                os.makedirs("%s/infered/%s/" % (self.save_path, subject_id))
            for i, pred in enumerate(y):
                np.save("%s/infered/%s/output_%s.npy" % (self.save_path, subject_id, i), pred)

        print("Done Inferring all test subjects")
