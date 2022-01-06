import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate
K = tf.keras.backend
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#### Loss functions ####

def dice_loss(y_true, y_pred):
    numerator = 2 * K.sum(y_true * y_pred, axis=-1)
    denominator = K.sum(y_true + y_pred, axis=-1)
    return 1 - (numerator / denominator)
    
def binary_ce(y_true, y_pred, sample_weight= None, label_smoothing = 0.1):
    #label_smoothing = tf.constant(label_smoothing, dtype = tf.float32)
    return K.mean(tf.keras.losses.BinaryCrossentropy(from_logits = True, label_smoothing = label_smoothing)(y_true, y_pred, sample_weight = sample_weight))

def categorical_ce(y_true, y_pred, sample_weight= None, label_smoothing = 0):
    return tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy(label_smoothing = label_smoothing, from_logits = True)(y_true, y_pred, sample_weight = sample_weight))
    #return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing = label_smoothing, from_logits = True)
#def weighted_categorical_ce(y_tre)


def celoss(y_true, y_pred):
    return K.mean(binary_ce(y_true, y_pred), axis = -1)

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def discriminator_loss(real, generated, fake_real = None):
    #here we smooth Ones label to avoid the disc from being to confident about real data
    real_loss = loss_obj(tf.ones_like(real) - tf.random.uniform(real.shape, minval = 0, maxval = 0.1), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    if fake_real is not None:
        #loss on real image from the other sample; Useful for DualGAN
        fake_real_loss = loss_obj(tf.zeros_like(fake_real), generated)
        total_disc_loss = real_loss + generated_loss + fake_real_loss
        return total_disc_loss / 3
    else:
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss / 2

def calc_l1_loss(real_image, cycled_image):
    #use boolean mask for loss computation
    mask = tf.math.greater_equal(real_image, tf.constant(-0.995, dtype = tf.float32))
    loss1 = tf.reduce_mean(tf.abs(tf.boolean_mask(real_image, mask) - tf.boolean_mask(cycled_image, mask)))
    return loss1

def masked_mse_loss(real, pred):
    mask = tf.math.greater_equal(real, tf.constant(-0.995, dtype = tf.float32))
    loss2 = tf.reduce_mean(tf.keras.losses.MSE(tf.boolean_mask(real, mask), tf.boolean_mask(pred, mask)))
    return loss2

def mse_loss(real_image, fake_image):
    loss2 = tf.reduce_mean(tf.keras.losses.MSE(real_image, fake_image))
    return loss2

def cgan_generator_loss(disc_generated_output, gen_output, target, LAMBDA = 100):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

def vae_sigmoid_loss_logits(pred, tar, z, mean, logvar):
    shape = tar.shape
    mask = tf.math.greater_equal(tar, tf.constant(0.001, dtype = tf.float32))
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=tar)
    #We put CE loss to 0 for background voxels
    cross_ent = tf.where(tf.math.greater_equal(tar, tf.constant(0.001, dtype = tf.float32)), cross_ent, 0)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = tf.reduce_sum(log_normal_pdf(z, 0., 0.), axis=[1, 2])
    logqz_x = tf.reduce_sum(log_normal_pdf(z, mean, logvar), axis=[1, 2])
    
    logpx_z, logpz, logqz_x = tf.cast(logpx_z, dtype = tf.float64), tf.cast(logpz, dtype = tf.float64), tf.cast(logqz_x, dtype = tf.float64)
    reduced_mean = tf.reduce_mean(logpx_z + logpz - logqz_x)
    return tf.cast(-reduced_mean, tf.float32)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def style_loss(pred, tar, pretrained_model):
    pred = Concatenate(axis = -1)([pred, pred, pred])
    tar = Concatenate(axis = -1)([tar, tar, tar])
    pred_output = pretrained_model(pred)
    tar_output = pretrained_model(tar)
    gram_pred = [gram_matrix(l) for l in pred_output]
    gram_tar = [gram_matrix(l) for l in tar_output]
    return tf.reduce_mean([tf.reduce_mean((gram_pred[i] - gram_tar[i])**2) for i in range(len(gram_pred))])

def content_loss(pred, tar, pretrained_model):
    pred = Concatenate(axis = -1)([pred, pred, pred])
    tar = Concatenate(axis = -1)([tar, tar, tar])
    pred_output = pretrained_model(pred)
    tar_output = pretrained_model(tar)
    return tf.reduce_mean([tf.reduce_mean((pred_output[i] - tar_output[i])**2) 
                            for i in range(len(pred_output))])

def ssim_loss(pred, tar):
    return 1 - tf.reduce_mean(tf.image.ssim(tar, pred, 1.0))

def contrastive_loss(z1, z2, y_true):
    margin = tf.constant(1.0E1)
    y_pred = tf.linalg.norm(z1 - z2, axis=1)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(tf.math.maximum(margin - y_pred, 0.0)))
    
LAMBDA = tf.constant(200, dtype = tf.float32)
LAMBDA2 = tf.constant(3, dtype = tf.float32)

############################# Training fct #################
@tf.function
def train_step(ClassModel, input_image, target, epoch, write = False):
    """
    Classical training step function
    @model : model class containing a model to train, its optimizer, its writer 
    @write : indicates if we log training info for tensorboard analysis
    @epoch : the epoch we are at 
    """
    with tf.GradientTape(persistent = True) as tape:
        output = ClassModel.model(input_image, training=True)
        # calculate the loss
        mse = mse_loss(target, output)
        
    # Calculate the gradients
    gradients = tape.gradient(mse, 
                              ClassModel.model.trainable_variables)
    # Apply the gradients to the optimizer
    ClassModel.optimizer.apply_gradients(zip(gradients, 
                                        ClassModel.model.trainable_variables))
    if write :
        with ClassModel.writer.as_default():
            tf.summary.scalar('loss', mse, step=epoch)

@tf.function
def val_step(ClassModel, input_image, target, epoch, on_cpu):
    """
    Classical training step function
    @model : model class containing a model to train, its optimizer, its writer
    @epoch : the epoch we are at 
    """
    with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
        output = ClassModel.model(input_image, training=False)
        # calculate the loss
        mse = mse_loss(target, output)
        with ClassModel.writer.as_default():
            tf.summary.scalar('val_loss', mse, step=epoch)
        return mse 
            
def train(ClassModel, epochs = 10, steps = 25, load_ckpt = False, val_on_cpu = False):
    """
        fct to train our model
    """
    checkpoint_path = ClassModel.save_path + "/checkpoints/train"
    ckpt = tf.train.Checkpoint(model=ClassModel.model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    # if a checkpoint exists, restore the latest checkpoint.
    try:
        if ckpt_manager.latest_checkpoint and load_ckpt:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
            for opt in ClassModel.optimizer:
                opt.lr.assign(ClassModel.initial_learning_rate)
    except:
        print("Could not restore latest checkpoint, continuing as if!")
    print("Start training")
    for X_val, Y_val in ClassModel.data_generator("val"):
        #we load val data only once
        for e in range(1, epochs):
            start = time.time()
            n = 1
            for image, target in ClassModel.data_generator():
                train_step(ClassModel, image, target, tf.constant(e, dtype = tf.int64), write = True)
                if n % steps == 0:
                    print ('.', end='')
                    break
                n += 1
            if e % 5 == 0:
                #validation every 5 epochs
                val_loss = val_step(ClassModel, X_val, X_val, tf.constant(e, dtype = tf.int64), val_on_cpu)
        break
    ClassModel.model.save_weights(ClassModel.save_path + '/best.h5')

###################  DualGAN functions ###################
@tf.function
def train_step_dualGAN(model, input_image_A, input_image_B, epoch, write = False):
    training = True
    with tf.GradientTape(persistent = True) as tape:
        gen_output_A = model.generator_A(input_image_B, training=training)
        recov_B = model.generator_B(gen_output_A, training=training)
            
        gen_output_B = model.generator_B(input_image_A, training=training)
        recov_A = model.generator_A(gen_output_B, training=training)
        
        disc_real_A = model.discriminator_A(input_image_A, training=training)
        disc_real_B = model.discriminator_B(input_image_B, training=training)

        disc_fake_A = model.discriminator_A(gen_output_A, training=training)
        disc_fake_B = model.discriminator_B(gen_output_B, training=training)

        #We add this metric so that the discriminators will learn to differentiate the 2 data samples (on real images this time)
        disc_fake_on_real_B = model.discriminator_B(input_image_A, training=training)
        disc_fake_on_real_A = model.discriminator_A(input_image_B, training=training)
        
        # calculate the loss
        #Here generator A tries to fool Disc A / same for B
        #We add another l1 loss minimizing the gap between both images as we know it is tiny 
        gen_A_loss = generator_loss(disc_fake_A)  + LAMBDA2 * calc_l1_loss(input_image_B, gen_output_A)
        gen_B_loss = generator_loss(disc_fake_B) + LAMBDA2 * calc_l1_loss(input_image_A, gen_output_B)
        
        total_cycle_loss = calc_l1_loss(input_image_A, recov_A) + calc_l1_loss(input_image_B, recov_B)
        
        # Total generator loss = adversarial loss + cycle loss
        total_gen_A_loss = gen_A_loss + LAMBDA * total_cycle_loss
        total_gen_B_loss = gen_B_loss + LAMBDA * total_cycle_loss
        
        disc_A_loss = discriminator_loss(disc_real_A, disc_fake_A)#, disc_fake_on_real_A)
        disc_B_loss = discriminator_loss(disc_real_B, disc_fake_B)#, disc_fake_on_real_B)
    # Calculate the gradients for generator and discriminator
    generator_A_gradients = tape.gradient(total_gen_A_loss, 
                                          model.generator_A.trainable_variables)
    generator_B_gradients = tape.gradient(total_gen_B_loss, 
                                          model.generator_B.trainable_variables)
    
    discriminator_A_gradients = tape.gradient(disc_A_loss, 
                                              model.discriminator_A.trainable_variables)
    discriminator_B_gradients = tape.gradient(disc_B_loss, 
                                              model.discriminator_B.trainable_variables)
    
    # Apply the gradients to the optimizer
    model.generator_A_optimizer.apply_gradients(zip(generator_A_gradients, 
                                                    model.generator_A.trainable_variables))
    model.generator_B_optimizer.apply_gradients(zip(generator_B_gradients, 
                                                    model.generator_B.trainable_variables))
    model.discriminator_A_optimizer.apply_gradients(zip(discriminator_A_gradients,
                                                        model.discriminator_A.trainable_variables))
    model.discriminator_B_optimizer.apply_gradients(zip(discriminator_B_gradients,
                                                        model.discriminator_B.trainable_variables))
    
    if write :
        with model.writer.as_default():
            tf.summary.scalar('total_gen_A_loss', total_gen_A_loss, step=epoch)
            tf.summary.scalar('total_gen_B_loss', total_gen_B_loss, step=epoch)
            tf.summary.scalar('disc_A_loss', disc_A_loss, step=epoch)
            tf.summary.scalar('disc_B_loss', disc_B_loss, step=epoch)
            tf.summary.scalar('GEN_A Learning rate', model.generator_A_optimizer.lr, step=epoch)
            tf.summary.scalar('Disc_A Learning rate', model.discriminator_A_optimizer.lr, step=epoch)
            tf.summary.scalar('GEN_B Learning rate', model.generator_B_optimizer.lr, step=epoch)
            tf.summary.scalar('Disc_B Learning rate', model.discriminator_B_optimizer.lr, step=epoch)
            tf.summary.scalar('total cycle loss', total_cycle_loss, step=epoch)
            for t in discriminator_A_gradients :
                tf.summary.histogram("disc_A_%s " % t.name, data=t, step = epoch)
        
        

@tf.function
def val_step_dualGAN(model, input_image_A, input_image_B, epoch, on_cpu = False):
    #Somethimes We want to run our validation on CPU, cause we usely dont have enough memory on GPU
    with tf.device('/device:%s:0' % "CPU" if on_cpu else "GPU"):
        training = False
        gen_output_A = model.generator_A(input_image_B, training=training)
        recov_B = model.generator_B(gen_output_A, training=training)
            
        gen_output_B = model.generator_B(input_image_A, training=training)
        recov_A = model.generator_A(gen_output_B, training=training)

        disc_real_B = model.discriminator_B(input_image_B, training=training)
        disc_real_A = model.discriminator_A(input_image_A, training=training)
        
        disc_fake_A = model.discriminator_A(gen_output_A, training=training)
        disc_fake_B = model.discriminator_B(gen_output_B, training=training)
        
        #We add this metric so that the discriminators will learn to differentiate the 2 data samples (on real images this time)
        # disc_fake_on_real_B = model.discriminator_B(input_image_A, training=training)
        # disc_fake_on_real_A = model.discriminator_A(input_image_B, training=training)
        
        # calculate the loss
        #Add l1 loss to avoid too big modification by Generators
        gen_A_loss = generator_loss(disc_fake_A)  + LAMBDA2 * calc_l1_loss(input_image_B, gen_output_A)
        gen_B_loss = generator_loss(disc_fake_B) + LAMBDA2 * calc_l1_loss(input_image_A, gen_output_B)
        
        total_cycle_loss = calc_l1_loss(input_image_A, recov_A) + calc_l1_loss(input_image_B, recov_B)
        # Total generator loss = adversarial loss + cycle loss
        total_gen_A_loss = gen_A_loss + total_cycle_loss * LAMBDA
        total_gen_B_loss = gen_B_loss + total_cycle_loss * LAMBDA
        
        disc_A_loss = discriminator_loss(disc_real_A, disc_fake_A)#, disc_fake_on_real_A)
        disc_B_loss = discriminator_loss(disc_real_B, disc_fake_B)#, disc_fake_on_real_B)
        with model.writer.as_default():
            tf.summary.scalar('total_gen_A_val_loss', total_gen_A_loss, step=epoch)
            tf.summary.scalar('total_gen_B_val_loss', total_gen_B_loss, step=epoch)
            tf.summary.scalar('disc_A_val_loss', disc_A_loss, step=epoch)
            tf.summary.scalar('disc_B_val_loss', disc_B_loss, step=epoch)
            tf.summary.scalar('Val total cycle loss', total_cycle_loss, step=epoch)
        return [total_gen_A_loss, total_gen_B_loss, disc_A_loss, disc_B_loss]


################################""

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=500):
        super(CustomSchedule, self).__init__()
        self.lr = 0
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        self.lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        return self.lr
class CustomExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, lr, decay_steps, decay_rate, staircase = True):
        super(CustomExponentialDecay, self).__init__(lr, decay_steps, decay_rate, staircase)
        self.lr = lr
        
    def __call__(self, step):
        self.lr = super(CustomExponentialDecay, self).__call__(step)
        return self.lr
    
class Scheduler():
    def __init__(self, list_optimizers, names, ratios = [1], patience = 10, decay_rate = 0.8, early_stopping = np.inf):
        #Infos saved as a dictionnary
        #opt : [min_val_loss, nb_epoch_without_improvment]
        self.training_infos = {}
        self.patiences = [patience * ratio for ratio in ratios]
        self.decay_rate = decay_rate
        self.total_loss = np.inf
        self.early_stopping = early_stopping
        self.early_stopping_counter = 0
        for i, opt in enumerate(list_optimizers) :
            self.training_infos[names[i]] = [np.inf, 0, opt]

    def update(self, losses):
        for i, name in enumerate(self.training_infos):
            infos_opt = self.training_infos[name]
            if losses[i].numpy() < infos_opt[0]:
                if name == "Disc" and (losses[i].numpy() < (infos_opt[0] * 0.66)):
                    #here we have a drastic Disc amelioration, its when "cops" becomes better than "thieves"
                    #we want Gen to start from that point now!
                    print("Drastic Discriminator amelioration! Gen callbacks reseted!")
                    self.total_loss += (losses[0].numpy() - self.training_infos["Gen"][0])
                    self.training_infos["Gen"][:2] = [losses[0].numpy(), 0]
                print("%s loss reduced from %s to %s" % (name, infos_opt[0], losses[i].numpy()))
                #patience reset to 0 and we update best loss
                self.training_infos[name][:2] = [losses[i].numpy(), 0]
            else:
                print("%s loss has not reduced from %s" % (name, infos_opt[0]))
                self.training_infos[name][1] += 1
                if self.training_infos[name][1] == self.patiences[i] :
                    self.training_infos[name][1] = 0
                    if "Disc" in name:
                        self.training_infos[name][0] = losses[i].numpy()
                        print("%s lr updated & best losses reseted" % name)
                    # else:
                    #     #here we reload best checkpoint model for Gen!
                    #     self.model.generator.load_weights(self.model.save_path + '/best_g.h5')
                    #     print("Generator Wweights reloaded from previous best checkpoint")
                    opt = self.training_infos[name][2]
                    try:
                        opt.lr.assign(opt.lr * self.decay_rate)
                    except:
                        try:
                            opt.lr.lr  *= self.decay_rate
                        except:
                            pass
                    print("Reducing learning_rate of %s optimizer to %s" % (name, opt.lr))
        total = 0
        for i, name in enumerate(self.training_infos) :
            print(name)
            if name not in ["Disc", "Modules"]:
                total += losses[i].numpy()
        if total <= self.total_loss:
            self.total_loss = total
            self.early_stopping_counter = 0
            return True
        self.early_stopping_counter += 1
        if self.early_stopping_counter == self.early_stopping :
            return "stop"
        return False
    
    ################################
    #My callbacks
    ##################################

class ModifiedReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self,
                 save_path,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        super(ModifiedReduceLROnPlateau, self).__init__(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr)
        self.save_path = save_path
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                            self.model.load_weights(self.save_path + "/best.h5")
                            print('Best model reloaded')

############### Load pretrained models #####################
def load_vgg(input_shape):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape + (3,))
    vgg.trainable=False
    contentLayers = ["block5_conv2"]
    styleLayers = ['block1_conv1',
                   'block2_conv1',
                   'block3_conv1', 
                   'block4_conv1', 
                   'block5_conv1']
    return tf.keras.Model(vgg.input, [vgg.get_layer(name).output for name in styleLayers])
        

                            
############################### TRANSFORMER #####################

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
