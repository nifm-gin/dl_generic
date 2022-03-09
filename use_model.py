import sys
import argparse
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152                                                                                                                                             
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments for patches creation.')
    parser.add_argument("-is", '--input_shape', type=int, nargs='+', default = (64,64),
                        help='Input Shape of model, by default 64*64*64')
    parser.add_argument("-ps", '--patch_shape', type=int, default = 16,
                        help='Patches Shape, by default 16. Only for transformer models')
    parser.add_argument("-dp", '--data_path', type=str, default = None,
                        help = "path to the directory containing the patches")
    parser.add_argument("-d", '--depth', type = int, default = 4,
                        help='Depth of our Unet model, by default 4')
    parser.add_argument("-a", '--augmentations', type=str, nargs='*', default = [],
                        help='Name of the augmentations functions to apply to patches')
    parser.add_argument("-s", '--strides', type = int, default = 2,
                        help='Strides used for each convolutional layer, by default (1, 1, 1)')
    parser.add_argument('--pool_size', type = int, nargs = '+', default = 2,
                        help='Size of each max pool layer, by default (2, 2, 2)')
    parser.add_argument("-fs", '--filter_size', type = int, nargs = '+', default = 3,
                        help='Depth of our Unet model, by default (3,3,3)')
    parser.add_argument("-bs", '--batch_size', type = int, default = 8,
                        help='Batch size required for training, by default 8')
    parser.add_argument('--burning_steps', type = int, default = 0,
                        help='Burning steps to run before training, default : 0')
    parser.add_argument("-lr", '--learning_rate', type = float, default = 0.001,
                        help='Learning rate required to start our training, 10E-3 by default.')
    parser.add_argument("-nc", '--num_channels', type = int, default = 1,
                        help='Number of contrasts as input, by default 1')
    parser.add_argument("-vs", '--vocabulary_size', type = int, default = 10000,
                        help='Size of vocabulary. For transformer models only.')
    parser.add_argument("-dm", '--d_model', type = int, default = 512,
                        help='Dimension of our model. For transformer models only.')
    parser.add_argument("-nh", '--num_heads', type = int, default = 8,
                        help='Number of heads in the multi-headed self attention layers. For transformer models only.')
    parser.add_argument("-e", '--epochs', type = int, default = 10,
                        help='Number of training epochs , by default 10')
    parser.add_argument("-ff", '--features_factor', type = int, default = 16,
                        help='Factor by which multiply the powers of 2, will give the features number needed at each depth level , by default 16')
    parser.add_argument("-spe", '--steps_per_epoch', type = int, default = 25,
                        help='Number of steps for each epoch , by default 25')
    parser.add_argument("-wp", '--weights_path', type=str, default = None,
                        help = "path to the file containing already trained model's weighs. None by default")
    parser.add_argument("-l", '--loss', type = str, default = "dice",
                        help='Loss used to train our segmentation unet model (dice by default), for generation : MAE is used')
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
    parser.add_argument('--load_ckpt', action = 'store_true',
                        help = "Indicates wether we want to restore the last training checkpoint or not")
    parser.add_argument("-cw", '--class_weights', type=float, nargs='+', default = None,
                        help='Class weights, if under-represented class')
    parser.add_argument("-sw", '--sequence_weights', type=float, nargs='+', default = None,
                        help='Sequence weights ratios, if under-represented sequence class')
    parser.add_argument('--loss_ratios', type=float, nargs='+', default = [1, 1,1, 1],
                        help='Loss ratios between modules losses for ImUnity training')
    parser.add_argument('--num_classes', type = int, default = 2,
                        help='Number of classes to predict, 2 by default')
    parser.add_argument('--num_sequences', type = int, default = 1,
                        help='Number of sequences in the dataset. ImUnity model will include a module assuring sequences particularities preservation. 1 by default')
    parser.add_argument('--latent_dim', type = int, default = 2,
                        help='Dimension of the latent space, useful when manipulating VAE models.')
    parser.add_argument('-dr','--decay_rate', type = float, default=None,
                        help = "Indicates the decay rate ratio we want to use exponential lr decay. Else no decay rate is used for learning rate.")
    parser.add_argument('--val_on_cpu', action = 'store_true',
                        help = "Indicates wether we want to run validation on cpus only or not. For RAM issues.")
    parser.add_argument('--model', type=str, default = "Unet",
                        help = "Which Model to use, by default : Unet.")
    parser.add_argument("-ai", '--additional_inputs', type = str, nargs="*", default = None, help = "list of clinical inputs to add")
    parser.add_argument('--name', type = str, default = "Model", help = "Model's name, useful for tensorboard lookups")
    parser.add_argument('--self_pretrain', action = 'store_true',
                        help = "Indicates wether we want to self-pretrain our  model using contrastive loss")
    parser.add_argument('--saliency', action = 'store_true',
                        help = "Indicates wether we want to run saliency at the  end of inference")
    parser.add_argument("-gtp", '--gt_path', type=str, default = None,
                        help = "path to the groudn truth csv")

    args = parser.parse_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if len(args.gpu) == 1 :
        os.environ["CUDA_VISIBLE_DEVICES"]="%s" % args.gpu[0]
        print("GPU %s in use" % args.gpu[0])
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    elif len(args.gpu) == 2:
        os.environ["CUDA_VISIBLE_DEVICES"]="%s,%s" % tuple(args.gpu)
        print("GPUs %s,%s in use" % tuple(args.gpu))
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        config = tf.config.experimental.set_memory_growth(physical_devices[1], True)
    
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        print("Using CPU only")
    if args.model == "CNN":
        from CNN import CNN
        model = CNN(input_shape = tuple(args.input_shape), data_path = args.data_path, gt_path = args.gt_path,
                    num_channels = args.num_channels, batch_norm = args.batch_norm,
                    pool_size = args.pool_size, filter_shape = args.filter_size, name = args.name,
                    batch_size = args.batch_size, initial_learning_rate = args.learning_rate,
                    strides = args.strides, depth = args.depth, weights_path = args.weights_path,
                    save_path = args.save_path, model_type = args.model_type, features_factor = args.features_factor,
                    num_classes = args.num_classes, class_weights = args.class_weights, dropout = 0.2,
                    additional_inputs = args.additional_inputs, data_augmentation = args.augmentations)
    elif args.model == "Unet":
        from Unet import Unet
        model = Unet(input_shape = tuple(args.input_shape), data_path = args.data_path,
                     num_channels = args.num_channels, batch_norm = args.batch_norm,
                     pool_size = args.pool_size, filter_shape = args.filter_size, name = args.name,
                     batch_size = args.batch_size, initial_learning_rate = args.learning_rate,
                     strides = args.strides, depth = args.depth, weights_path = args.weights_path,
                     save_path = args.save_path, model_type = args.model_type, features_factor = args.features_factor,
                     loss = args.loss, data_augmentation = args.augmentations)
        
    elif args.model == "Resnet":
        from Resnet import Resnet
        model = Resnet(input_shape = tuple(args.input_shape), data_path = args.data_path, gt_path = args.gt_path,
                       num_channels = args.num_channels, batch_norm = args.batch_norm,
                       pool_size = args.pool_size, filter_shape = args.filter_size, name = args.name,
                       batch_size = args.batch_size, initial_learning_rate = args.learning_rate,
                       strides = args.strides, depth = args.depth, weights_path = args.weights_path,
                       save_path = args.save_path, model_type = args.model_type, features_factor = args.features_factor,
                       num_classes = args.num_classes, class_weights = args.class_weights, dropout = 0.2,
                       additional_inputs = args.additional_inputs, data_augmentation = args.augmentations)
    elif args.model == "ImUnity":
        from ImUnity import ImUnity
        model = ImUnity(input_shape = tuple(args.input_shape), data_path = args.data_path,
                        num_channels = args.num_channels, batch_norm = args.batch_norm,
                        pool_size = args.pool_size, filter_shape = args.filter_size, name = args.name,
                        batch_size = args.batch_size, initial_learning_rate = args.learning_rate,
                        strides = args.strides, depth = args.depth, weights_path = args.weights_path,
                        save_path = args.save_path, model_type = args.model_type,
                        features_factor = args.features_factor, num_sites = args.num_classes,
                        dropout = 0.2, decay_rate = args.decay_rate, latent_dim = args.latent_dim,
                        class_weights = args.class_weights, data_augmentation = args.augmentations,
                        loss_ratios = args.loss_ratios, num_sequences = args.num_sequences,
                        sequence_weights =args.sequence_weights, biological_features = args.additional_inputs)
    elif args.model == "GAN":
        from GAN import CGAN
        model = CGAN(input_shape = tuple(args.input_shape), data_path = args.data_path,
                     num_channels = args.num_channels, batch_norm = args.batch_norm,
                     pool_size = args.pool_size, filter_shape = args.filter_size, name = args.name,
                     batch_size = args.batch_size, model_type = "pix2pix",
                     initial_learning_rate = args.learning_rate,
                     strides = args.strides, depth = args.depth, weights_path = args.weights_path,
                     save_path = args.save_path, features_factor = args.features_factor)

    elif args.model == "VAE":
        from VAE import VAE
        model = VAE(input_shape = tuple(args.input_shape), data_path = args.data_path,
                    num_channels = args.num_channels, batch_norm = args.batch_norm,
                    pool_size = args.pool_size, filter_shape = args.filter_size, name = args.name,
                    batch_size = args.batch_size, model_type = "vae",
                    initial_learning_rate = args.learning_rate,
                    strides = args.strides, depth = args.depth, weights_path = args.weights_path,
                    save_path = args.save_path, features_factor = args.features_factor,
                    latent_dim = args.latent_dim)

    elif args.model == "VAEGAN":
        from VAEGAN import VAEGAN
        model = VAEGAN(input_shape = tuple(args.input_shape), data_path = args.data_path,
                       num_channels = args.num_channels, batch_norm = args.batch_norm,
                       pool_size = args.pool_size, filter_shape = args.filter_size, name = args.name,
                       batch_size = args.batch_size, model_type = "GAN",
                       initial_learning_rate = args.learning_rate,
                       strides = args.strides, depth = args.depth, weights_path = args.weights_path,
                       save_path = args.save_path, features_factor = args.features_factor,
                       latent_dim = args.latent_dim)

    else:
        from ClassificationTransformer import ClassificationTransformer
        model = ClassificationTransformer(input_shape = tuple(args.input_shape), data_path = args.data_path, patch_shape = args.patch_shape,
                                          num_channels = args.num_channels, dropout = 0.2, num_heads = args.num_heads,
                                          batch_size = args.batch_size, initial_learning_rate = args.learning_rate,
                                          depth = args.depth, weights_path = args.weights_path,
                                          save_path = args.save_path, model_type = args.model_type,
                                          features_factor = args.features_factor,
                                          d_model = args.d_model, vocabulary_size = args.vocabulary_size, num_classes = args.num_classes)
    if args.load_ckpt:
        model.restore_ckpt()
    if args.self_pretrain:
        model.self_pretrain(2000, args.steps_per_epoch, args.val_on_cpu)
    if not args.infer_only:
        # if args.self_pretrain:
        #     try:
        #         for layer in model.generator.layers[:2]:
        #             layer.trainable = False
        #     except:
        #         for layer in model.model.layers[:2]:
        #             layer.trainable = False
        if  args.burning_steps > 0:
            model.burn_steps(args.burning_steps)
        model.train(args.epochs, args.val_on_cpu)
        model.model.save_weights(model.save_path + '/final_model_weights.h5')
        print("Models weights saved.")
    if not args.train_only:
        model.infer(saliency  = args.saliency)
        print("Inference done!")
    print("Exiting script")
