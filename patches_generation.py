import nibabel as nib
import time
import os
import numpy as np
import sys
import argparse
from augmentations import *
import multiprocessing
from multiprocessing import Process, Pool
import pandas as pd
import pickle

root_dir = "/home/stenzel/Documents/dl_generic/"

def rescale_data(scan, mask_file  = False, sigmoid = False, segmentation = False):
    shape = scan.shape
    dim = len(shape) - 1#last dim is the number of sequences
    for i in range(shape[-1] - (1 * mask_file +1 * segmentation)):
        #we iterate over the sequences but not the mask if there is one or the output sequence in case of segmentation
        if dim == 2:
            M = np.quantile(scan[:, :, i], 0.99945)
            m = np.amin(scan[:, :, i])
            scan[:, :, i] = 2 * ((scan[:, :, i] - m) / (M - m) - 0.5)
            scan[:, :, i][scan[:, :, i] > 1] = 1
        else:
            M = np.quantile(scan[:, :, :, i], 0.99945)
            m = np.amin(scan[:, :, :, i])
            scan[:, :, :, i] = 2 * ((scan[:, :, :, i] - m) / (M - m) - 0.5)
            scan[:, :, :, i][scan[:, :, :, i] > 1] = 1
    if sigmoid:
        if mask_file or segmentation:
            scan[: -(1 * mask_file +1 * segmentation)]= (scan[: -(1 * mask_file +1 * segmentation)] + 1) / 2
        else:
            scan = (scan + 1) / 2
    return scan
    
def find_affine_space(data_path, a_contrast, save_path):
    """
    function saving the affine space array of our input data
    we will need it at the end of the pipeline to process the final output (if image to image task)
    in fact we want our output to have the same orientation as our input
    return input shape
    """
    for patient_ID in os.listdir(data_path + "/train"):
        #bids_folder = "dwi" if "dwi" in a_contrast or "DWI" in a_contrast else "anat"
        bids_folder = "anat"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        try:
            img = nib.load("%s/train/%s/%s/%s_%s.nii.gz" % (data_path, patient_ID, bids_folder, patient_ID, a_contrast))
        except FileNotFoundError:
            img  = nib.load("%s/train/%s/%s/%s_%s.nii" % (data_path, patient_ID, bids_folder, patient_ID, a_contrast))
        except nib.filebasedimages.ImageFileError:
            img = nib.load("%s/train/%s/%s/%s_%s.nii" % (data_path, patient_ID, bids_folder, patient_ID, a_contrast))
        np.save(save_path + "affine.npy", img.affine)
        return img.get_fdata().shape
            
def find_indices(patch_size, input_shape, overlap = 1., save_path = root_dir):
    """
    Function that will compute the indices needed for the patch extraction
    Will save them as an npy file so that it does not need to be done everytime
    3D only for the moment

    Args :
    - patch_size : required size of the patches
    - input_shape : shape of each scan
    - overlap : percentage value of overlap needed between patches
    - save_path : folder path for the generated patches to be stored in
    """
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    x_shape, y_shape, z_shape = input_shape
    #We first need to extract the indices for each patch
    #It will bbe easier later to work directly knowing indices
    indices = []
    ix, iy, iz = 0, 0, 0
    #Window indices that will slides across the whole array
    #Tricky cause we need to handle edge effect as input dim is not always a multiple of patch_dim
    while ix <= (x_shape - patch_size[0]):
        iy = 0
        while iy <= (y_shape - patch_size[1]):
            iz = 0
            while iz <= (z_shape - patch_size[2]):
                indices.append((ix, ix +patch_size[0], iy, iy + patch_size[1], iz , iz + patch_size[2]))
                iz += max(int(patch_size[2] * overlap), 1)
                
                if iz >= z_shape - patch_size[2]:
                    indices.append((ix, ix +patch_size[0], iy, iy + patch_size[1], z_shape - patch_size[2], z_shape))
                    break
            iy += int(patch_size[1] * overlap)
            if iy >= y_shape - patch_size[1]:
                if iy == y_shape - patch_size[1] + int(patch_size[1] * overlap):
                    break
                iy = y_shape - patch_size[1]
        
        ix += int(patch_size[0] * overlap)
        if ix >= x_shape - patch_size[0]:
            if ix == x_shape - patch_size[0] + int(patch_size[0] * overlap):
                break
            ix = x_shape - patch_size[0]

    #Save indexes to the path specified as a numpy file
    indices = np.asarray(indices)
    if save_path is not None:
        np.save(save_path + "patches_indices.npy", indices)
    return indices

def generate_patches(data, indices, augmentations, subjects, save_folder, output, mask_count = 0, classification = None, mask = False, additional_inputs = None, sigmoid = False, segmentation = False, style = False, affine_space = None):
    """
    Function that generates the patches and save them as npy file

    Args :
    - data : list scans to be processed
    - indices : The indices for patches extraction
    - augmentations : list of functions to appply as a data augmentation
    - subjects : list of subects id, useful for pretraining model on contrastive loss
    - save_folder : path to the folder where to save the patches
    - end_to_end : indicates the output type, can be a list of contrast, if None will not be an image-to-image task
    - data_type : train / val / test
    - classification : list of corresponding classes to data
    - mask : boolean telling us if we added the ROI mask as last channel
    - additional_inputs : list of additional_inputs matching data
    - sigmoid : indicates if data are scaled btw 0 1. -1 1 by default
    - segmentation : indicates wether or not patches will be for segmentation task. If True then we only keep patches with at least 1 ROI voxel
    - style : Indicates if we want to use some data as a style reference. Must be contained in a folder 'style' next to train/val/test ones
    - affine_space : ifnot None we  save it for  reconstruction : test subjects
    Requierements :
    - All data is supposed to have similar shape
    
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if affine_space is not None:
        np.save("%s/affine_space.npy" % save_folder, affine_space)
    cpt = 0
    start = time.time()
    for augmentation in augmentations:
        #with Pool(multiprocessing.cpu_count()) as p:
        data += [augmentation(scan) for scan in data]
        if classification != None:
            classification += classification
        if additional_inputs is not None:
            additional_inputs += additional_inputs
        subjects += subjects
    input_pos_end = - len(output) - 1*style + -1 * mask
    if input_pos_end == 0:
        input_pos_end = None
    current_sub, sub_start = subjects[0], 0
    sub_dict = {}
    for i, scan in enumerate(data):
        if current_sub != subjects[i]:
            sub_dict[current_sub] = (sub_start, cpt)
            current_sub = subjects[i]
            sub_start = cpt
        m = np.amin(scan[:, :, :, 0])
        #we iterate over all scans
        for (xs, xf, ys, yf, zs, zf) in indices:
            #we extract patches from scans
            if np.count_nonzero(scan[xs : xf, ys : yf, zs : zf, 0] != m) >= mask_count :
                #we count the number of brain voxels, m being the background value
                #mask is  a 0/1 sequence
                if mask and (np.sum(scan[xs : xf, ys : yf, zs : zf, -1]) <= mask_count):
                    #we skip the patch if not enough ROI voxels in it
                    #if zero ROI voxels or less than mask count then we skip the patch for the scan
                    continue
                if segmentation and (np.sum(scan[xs : xf, ys : yf, zs : zf, -1 - (1 * mask)]) == 0):
                    #Skipping patches with no ROI voxel
                    continue
                if output:
                    np.save("%s/output_%s.npy" % (save_folder, cpt), scan[xs : xf, ys : yf, zs : zf, -len(output) - (1 * mask):scan.shape[-1] - (1 * mask)])
                    # Contrast / segmentation generation
                elif style:
                    #We save style as the output contrast, not possible to specify output sequence and style generation at the same time
                    np.save("%s/output_%s.npy" % (save_folder, cpt), scan[xs : xf, ys : yf, zs : zf, -( 1* style + 1*mask)])
                
                if classification is None:
                    if additional_inputs is not None:
                        np.savez("%s/%s.npz" % (save_folder, cpt), data = scan[xs : xf, ys : yf, zs : zf, 0 : input_pos_end],
                                 add_in = additional_inputs[i], sub = subjects[i])
                    else:
                        np.savez("%s/%s.npz" % (save_folder, cpt), data =  scan[xs : xf, ys : yf, zs : zf, 0 : input_pos_end], sub = subjects[i])
                        #useful for test data
                else:
                    #classification task
                    if additional_inputs is not None:
                        np.savez("%s/%s.npz" % (save_folder, cpt), data = scan[xs : xf, ys : yf, zs : zf,: input_pos_end], group = classification[i],
                                 add_in = additional_inputs[i], sub = subjects[i])
                    else:
                        np.savez("%s/%s.npz" % (save_folder, cpt), data = scan[xs : xf, ys : yf, zs : zf,: input_pos_end], group = classification[i], sub = subjects[i])
                cpt += 1
    sub_dict[current_sub] = (sub_start, cpt)
    with open("%s/subjects_patches.pkl" % save_folder, 'wb') as f:
        pickle.dump(sub_dict, f)
    np.save("%s/nb_patches.npy" % save_folder, cpt)
    return

def patches_generation(data_path, patch_size, input_shape, contrasts, overlap = 1., indices_path = None, augmentations = [],
                       output = [], save_path = root_dir + "patches/", mask_count = 0, nii_only = False, classification_outputs = None,
                       rescale = None, mask_file = None, additional_inputs = None, segmentation = False, style = False, sequential = False,
                       mask_on_test = False, transpose_axis = None):
    """
    Function that will extract patches from given data
    3D for the moment

    Args :
    - data_path : path to the subjects folder, according to BIDS norm  (https://bids.neuroimaging.io/) 
    - patch_size : required size of the patches, as a list
    - input_shape : shape of each scan
    - contrasts : A list of different contrasts to take into account, they will then be concatenated to 1 patches with several channels; 
    - overlap : percentage value of overlap needed between patches
    - indices_path : path to indices file previously computed
    - indices : If given, will be the indices for patches extraction
    - augmentations : list of functions to appply as a data augmentation, none by default
    - output : indicates the output type, can be a list of contrast, if empty list : will not be an image-to-image task
    - save_path : path to the folder where to save patches
    - mask_count : int, optional. Indicates the min number of brain voxels to be involved in each patch
    - mask_file : region of interest, specify the mask name as if it was a sequence / contrast
    - classification_outputs : array of clinical outputs for our model, they must be located in a "classification_outputs.csv" file in the data_path folder
    - additional_inputs : array of clinical inputs used as inputs of our model, they must be located in a "additional_inputs.csv" file in the data_path folder
    - style : Indicates if we want to use some data as a style reference. Must be contained in a folder 'style' next to train/val/test ones
    - segmentation : if output is segmentation mask we do not want to rescale it
    - transpose_axis : in case  of an axis_combination approach we  want  to transpose  our data to extract patches correctly
    Requierements :
    - All data is supposed to have similar shape
    """
    #First check data paths are correct
    try :
        assert os.path.isdir(data_path), "Data path specified does not exist"
        assert os.path.isdir(data_path + "/train/"), "Data path specified does not contain train data"
        assert os.path.isdir(data_path + "/val/"), "Data path specified does not contain val data"
        assert os.path.isdir(data_path + "/test/"), "Data path specified does not contain test data"
        if indices_path is not None:
            assert os.path.isfile(indices_path), "Indices path specified does not exist"
        if style :
            assert os.path.isdir(data_path + "/style/"), "Style folder path does not exist"
        if classification_outputs is not None:
            assert os.path.isfile(data_path + "/classification_outputs.csv"), "Classification csv path specified does not exist"
        if additional_inputs is not None:
            assert os.path.isfile(data_path + "/additional_inputs.csv"), "Additional_inputs.csv could not be find! Verify its location"
        if transpose_axis is not None:
            assert 1 in patch_size, "patch dimension  should be 2 as axis_combination is considered here"
    except AssertionError as err:
        sys.exit(err)
    
    if indices_path is not None:
        try:
            indices = np.load(indices_path)
            print("Indices loaded")
        except Exception as err:
            print("Failed to load indices", err)
    else:
        indices = find_indices(patch_size, input_shape, overlap, save_path)
        print("Indices computed")
    classes = None

    if additional_inputs is not None:
        dict_add_inputs = pd.read_csv(data_path + "/additional_inputs.csv", index_col = ["Patient"]).to_dict("index")
        
    if classification_outputs is not None:
        classes = pd.read_csv(data_path + "/classification_outputs.csv", index_col = ["Patient"]).to_dict("index")
    style_subjects = None
    if style :
        style_subjects = os.listdir("%s/style/" % data_path)
    procs = []
    for data_type in ["train", "val", "test"]:
        #Can be parallelized
        scans, groups, add_inputs, subjects = [], [], [], []
        for patient_ID in os.listdir(data_path + data_type):
            scan = np.zeros((*input_shape, len(contrasts + output) + ( 1 * style)+ (1 * (mask_file != None))))
            if style :
                #Here we load ROI mask as the last channel
                style_sub = random.choice(style_subjects)
                try:
                    if transpose_axis is not None:
                        scan[:, :, :, -(1 * style + (mask_file != None))]= np.transpose(nib.load("%s/style/%s/anat/%s_%s.nii.gz" % (data_path, style_sub, style_sub, contrasts[0])).get_fdata(), transpose_axis)
                    else:
                        scan[:, :, :, -(1 * style + (mask_file != None))]= nib.load("%s/style/%s/anat/%s_%s.nii.gz" % (data_path, style_sub, style_sub, contrasts[0])).get_fdata()
                except Exception as err:
                    print("Failed to open style file for subject : %s. \n Skipping patient" % (style_sub), err)
                    continue
            if mask_file != None:
                #Here we load ROI mask as the last channel
                try:
                    if transpose_axis is not None:
                        scan[:, :, :, -1]= np.transpose(nib.load("%s/%s/%s/anat/%s_%s.nii.gz" % (data_path, data_type, patient_ID, patient_ID, mask_file)).get_fdata(), transpose_axis)
                    else:
                        scan[:, :, :, -1]= nib.load("%s/%s/%s/anat/%s_%s.nii.gz" % (data_path, data_type, patient_ID, patient_ID, mask_file)).get_fdata()
                except Exception as err:
                    scan[:, :, :, -1] = 1
                    print("Failed to open mask file for subject : %s. All brain taken as ROI" % (patient_ID), err)
            if additional_inputs is not None:
                add_in = [dict_add_inputs[patient_ID][input_name] for input_name in additional_inputs]
            else:
                add_in = None
            data = None
            for i, contrast in enumerate(contrasts + output):
                #bids_folder = "dwi" if "dwi" in contrast or "DWI" in contrast else "anat"
                bids_folder = "anat"
                #Diffusion folder
                if contrast in output and data_type == "test":
                    continue
                try :
                    #We concatenate each contrast as an additionnal channel of our scan
                    data = nib.load("%s/%s/%s/%s/%s_%s.nii.gz" % (data_path, data_type, patient_ID, bids_folder, patient_ID, contrast))
                except FileNotFoundError:
                    try:
                        data = nib.load("%s/%s/%s/%s/%s_%s.nii" % (data_path, data_type, patient_ID, bids_folder, patient_ID, contrast))
                        print("Loaded .nii data for patient : %s" % patient_ID)
                    except Exception as err:
                        print("Failed to open %s file for subject : %s. Skipping this one" % (contrast, patient_ID), err)
                        break
                except nib.filebasedimages.ImageFileError:
                    try :
                        data = nib.load("%s/%s/%s/%s/%s_%s.nii" % (data_path, data_type, patient_ID, bids_folder, patient_ID, contrast))
                    except Exception as err:
                        print("Failed to open %s file for subject : %s. Skipping this one" % (contrast, patient_ID), err)
                        break
                except Exception as err:
                    print("Failed to open %s file for subject : %s. Skipping this one" % (contrast, patient_ID), err)
                    break
                if transpose_axis is not None:
                    scan[:, :, :, i] = np.transpose(data.get_fdata(), transpose_axis)
                else:
                    scan[:, :, :, i] = data.get_fdata()
                affine_space = data.affine
            if rescale is not None:
                scan = rescale_data(scan, mask_file != None, sigmoid = (rescale == "sigmoid"), segmentation = segmentation)
            if data_type == "test":
                if sequential:
                    generate_patches([scan], indices, [], [patient_ID], "%s/test/%s/" % (save_path, patient_ID), output, mask_count * mask_on_test, None, mask_file != None,
                                     [add_in] if add_in != None else None, False, False, style, affine_space)
                else:
                    p = Process(target = generate_patches, args = ([scan], indices, [], [patient_ID],
                                                                   "%s/test/%s/" % (save_path, patient_ID), output, mask_count * mask_on_test, None, mask_file != None,
                                                                   [add_in] if add_in != None else None, False, False, style, affine_space))
                    #for test data we take all slices without worrying about background voxels : mask_count = 0
                    p.start()
                    procs.append(p)
            
            else :
                if classes is not None:
                    try:
                        groups.append({class_name : classes[patient_ID][class_name] for class_name in classification_outputs})
                    except Exception as err:
                        print(err, "Skipping this subject")
                        continue
                if additional_inputs is not None :
                    add_inputs.append(add_in)
                scans.append(scan)
                try:
                    subjects.append(float(patient_ID.split("-")[1]))
                except IndexError:
                    subjects.append(float(patient_ID.split("_")[0][3:]))
                except ValueError:
                    try:
                        subjects.append(float(patient_ID[-4:]))
                    except:
                        subjects.append(float(patient_ID.split("_")[0][-1]))
        print("subjects list : ", subjects)
        if data_type in ["train", "val"]:
            p = Process(target = generate_patches, args = (scans, indices, augmentations if data_type == "train" else [], subjects,
                                                           save_path + data_type, output, mask_count, groups if classes is not None else None,
                                                           mask_file != None, add_inputs if additional_inputs != None else None,
                                                           rescale == "sigmoid", segmentation, style))
            p.start()
            procs.append(p)
    for p in procs:
        p.join()
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments for patches creation.')
    parser.add_argument("-ps", '--patch_shape', type=int, nargs=3, default = [64, 64, 64],
                        help='Shape of patches')
    parser.add_argument("-o", '--overlap', type=int, default = 0,
                        help='Overlap (in percent) required between patches')
    parser.add_argument("-dp", '--data_path', type=str,
                        help='path to thed dataset divided in train/val/test each containing subjects folder, according to BIDS norm')
    parser.add_argument("-ip", '--indices_path', type=str, default = None,
                        help='path to the npy file containing precomputed indices')
    parser.add_argument("-c", '--contrasts', type=str, nargs='+', default = ["T1"],
                        help='Name of the contrasts to take into account, as input')
    parser.add_argument("-out", '--outputs', type=str, nargs='+', default = [],
                        help='Name of the contrasts to take into account, as output.\n None by default')
    parser.add_argument("-a", '--augmentations', type=str, nargs='*', default = [],
                        help='Name of the augmentations functions to apply to each scan')
    parser.add_argument("-sp", '--save_path', type=str,
                        help='path to the folder where to save patches')
    parser.add_argument("-mc", '--mask_count', type=int, default = 0,
                        help='Min brain voxels in each patch. 0 by default')
    parser.add_argument("-mf", '--mask_file', type=str, default = None,
                        help='region of interest, specify the mask name as if it was a sequence / contrast')
    parser.add_argument('--nii_only', action = 'store_true',
                        help = "Indicates wether scan are saved as .nii instead of .nii.gz")
    parser.add_argument("-co", '--classification_outputs', type = str, nargs="*", default = None, help = "list of clinical outputs to predict")
    parser.add_argument('--rescale', type=str, default = None,
                        help = "If data needs to be rescaled, None by default, otherwise rescaling between -1 and 1 or 0 and 1 if arg==sigmoid")
    parser.add_argument("-ai", '--additional_inputs', type = str, nargs="*", default = None, help = "list of clinical inputs to add")
    parser.add_argument('--segmentation', action = 'store_true',
                        help = "Indicates wether output sequence is a segmentation mask or not")
    parser.add_argument('--style', action = 'store_true',
                        help = "Indicates if we want to use some data as a style reference. Must be contained in a folder 'style' next to train/val/test ones")
    parser.add_argument('--sequential', action = 'store_true',
                        help = "Indicates wether we want to sequencialize the generation in the case where the  RAM is saturated")
    parser.add_argument('--mask_on_test', action = 'store_true',
                        help = "Indicates wether we want to run inference on masked data only or not")
    parser.add_argument("-ac", '--axis_combination', action = 'store_true',
                        help = "Indicates wether we want to generate patches  along 3axis, for a 2.5d/3Dfusion approach")


    args = parser.parse_args()

    #find affine space of data, supposed to be the same for all data present in the dataset
    input_shape = find_affine_space(args.data_path, args.contrasts[0], args.save_path)
    overlap = 1. - (args.overlap / 100)
                
    patches_generation(args.data_path, args.patch_shape, input_shape, args.contrasts, overlap = overlap,
                       indices_path = args.indices_path, augmentations = [dict_augmentations[name] for name in args.augmentations],
                       output = args.outputs, save_path = args.save_path, mask_count = args.mask_count,
                       nii_only = args.nii_only, classification_outputs = args.classification_outputs, rescale = args.rescale,
                       mask_file = args.mask_file, additional_inputs = args.additional_inputs, segmentation = args.segmentation, style = args.style, sequential = args.sequential,
                       mask_on_test = args.mask_on_test)
    
    if args.axis_combination:
        assert 1 in args.patch_shape, "2.5D approach only makes  sense when using 2d patches. Patches  dimensions required here : %s" % patch_size
        initial_axis = 0 if args.patch_shape[0] == 1 else 1 if args.patch_shape[1] == 0 else 2
        patch_ratio = np.prod([i for i in args.patch_shape]) / np.prod([i if args.patch_shape[j] > 1 else 1 for j, i in enumerate(input_shape)])
        transpose_dict = {0 : (1,2,0),
                          1 : (0, 2, 1),
                          2 : (0, 1, 2)}
        for j, i in enumerate(args.patch_shape):
            if i > 1:
                _ = find_affine_space(args.data_path, args.contrasts[0], args.save_path[:-1] + "_axis_%s/" % j)
    
                #on transpose le patch:
                new_patch_shape = [m for m in args.patch_shape]
                new_patch_shape[j] = 1
                new_patch_shape[initial_axis] = int(patch_ratio * np.prod([m if j != n else 1 for n, m in enumerate(input_shape)]) / np.prod([m for m in new_patch_shape]))
                new_patch_shape = [new_patch_shape[n] for n in transpose_dict[j]]
                new_input_shape = tuple([input_shape[n] for n in transpose_dict[j]])
                print(new_patch_shape, new_input_shape)
                patches_generation(args.data_path, new_patch_shape, new_input_shape, args.contrasts, overlap = overlap,
                                   indices_path = None, augmentations = [dict_augmentations[name] for name in args.augmentations],
                                   output = args.outputs, save_path = args.save_path[:-1] + "_axis_%s/" % j, mask_count = args.mask_count,
                                   nii_only = args.nii_only, classification_outputs = args.classification_outputs, rescale = args.rescale,
                                   mask_file = args.mask_file, additional_inputs = args.additional_inputs, segmentation = args.segmentation, style = args.style, sequential = args.sequential,
                                   mask_on_test = args.mask_on_test, transpose_axis =  transpose_dict[j])         
