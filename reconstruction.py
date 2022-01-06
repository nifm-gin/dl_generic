import nibabel as nib
import os
import numpy as np
import sys
import argparse
from multiprocessing import Process

def image_reconstruction(data_path, subject_id, output_shape, indices_path, save_path = "home/stenzel/Documents/", affine_path = "home/stenzel/Documents/patches/affine.npy", saliency = False):
    """
    Function that will create the final image from a list of patches (here we only have the path to these patches)
    Mainly useful for inference

    Args :
    - data_path : path to the patches to be processed 
    - output_shape : shape of the output
    - indices_path : path to indices file previously computed
    - save_path : path to the folder where to save output
    
    Requierements:
    - All patches to be created and stored in the same folder as npy files
    - patch's format : output_PATCHNB.npy, PATCHNB is the positions of the patch in the indices. Is automated
    """
    data_path += subject_id
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    indices = np.load(indices_path + "patches_indices.npy")
    mask = np.zeros(output_shape)
    output = np.zeros(output_shape)
    contrast = "saliency" if saliency else "output"
    for i, (xs, xf, ys, yf, zs, zf) in enumerate(indices):
        mask[xs:xf, ys:yf, zs:zf] += 1
        scan = np.load("%s/%s_%s.npy" % (data_path, contrast, i))
        #scan = np.load("%s/%s.npy" % (data_path,  i))
        if scan.ndim == 3:
            #2D patches
            output[xs:xf, ys:yf, zs:zf] = output[xs:xf, ys:yf, zs:zf] + scan
        else :
            #3D patches
            output[xs:xf, ys:yf, zs:zf] = output[xs:xf, ys:yf, zs:zf] + scan.reshape(scan.shape[:-1])
    #would be problematic...
    mask[mask == 0] = 1
    output = np.divide(output, mask)
    affine = np.load(affine_path  + "affine_space.npy")
    output = nib.Nifti1Image(output, affine)
    output.to_filename(os.path.join(save_path, '%s_model_%s.nii.gz' % (subject_id,contrast)))
                       
def combine_results(data_path, subject_id):
    transpose_dict = {
        0 : (2,0,1),
        1 : (0,2,1),
        2 : (0,1,2)}
    res = nib.load("%s/results/%s/%s_model_output.nii.gz" % (data_path, subject_id, subject_id))
    affine =  res.affine
    res = res.get_fdata()
    res0 = np.transpose(nib.load("%s_axis0/results/%s/%s_model_output.nii.gz" % (data_path[:-1], subject_id, subject_id)).get_fdata(), transpose_dict[0])
    res1 = np.transpose(nib.load("%s_axis1/results/%s/%s_model_output.nii.gz" % (data_path[:-1], subject_id, subject_id)).get_fdata(), transpose_dict[1])
    all_res = np.stack((res, res0, res1), axis = -1)
    output = np.median(all_res, axis = -1)
    output = nib.Nifti1Image(output, affine)
    output.to_filename("%s/results/%s/%s_model_output_combination.nii.gz" % (data_path, subject_id, subject_id))
    output = nib.Nifti1Image(res0, affine)
    output.to_filename("%s/results/%s/%s_axis0_model_output.nii.gz" % (data_path, subject_id, subject_id))
    output = nib.Nifti1Image(res1, affine)
    output.to_filename("%s/results/%s/%s_axis1_model_output.nii.gz" % (data_path, subject_id, subject_id))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments for patches creation.')
    parser.add_argument('-o', '--output_shape', type=int, nargs=3, default = [256, 256, 192],
                        help='Output shape')
    parser.add_argument('-dp', '--data_path', type=str, default = "/home/stenzel/Documents/patches/infered/",
                        help='root path to the infered patches folders, divided in subject_id infered patches')
    parser.add_argument('-ip', '--indices_path', type=str, default = "/home/stenzel/Documents/patches/patches_indices.npy",
                        help='path to the npy file containing precomputed indices')
    parser.add_argument('-ap', '--affine_path', type=str, default = "/home/stenzel/Documents/patches/affine.npy",
                        help='path to the npy file containing affine space of input data')
    parser.add_argument('-sp', '--save_path', type=str, default = "/home/stenzel/Documents/patches/results/",
                        help='path to the folder where to save output')
    parser.add_argument('--saliency', action = 'store_true',
                        help = "Reconstruct saliency map from saliency patches")
    parser.add_argument('--axis_combination', action = 'store_true',
                        help = "In case we had a 2.5D approach and had a model for each axis")
    parser.add_argument('--process_lim', type=int, default = 100,
                        help='Number of processes max to run in parallel')
    args = parser.parse_args()
    procs = []
    cpt = 0
    data_path = args.data_path
    for subject_id in os.listdir(args.data_path):
        if os.path.isdir(args.data_path + subject_id):
            p = Process(target = image_reconstruction, args = (args.data_path, subject_id, args.output_shape, args.indices_path,
                                                               "%s/results/%s" % (args.save_path, subject_id), "%s/test/%s/" % (args.affine_path, subject_id), args.saliency))
            p.start()
            procs.append(p);cpt+=1
            if cpt % args.process_lim == 0:
                for p in procs:
                    p.join()
                procs = []
    for p in procs:
        p.join()
    procs = []; cpt = 0
    
    if args.axis_combination:
        transpose_dict = {0 : (1,2,0),
                          1 : (0, 2, 1),
                          2 : (0, 1, 2)}
        for i in [0,1]:
            data_path = "/".join(args.data_path.split("/")[:-2]) + "_axis%s/infered/" % i
            output_shape  = tuple([args.output_shape[n] for n in transpose_dict[i]])
            indices_path = args.indices_path[:-1] + "_axis_%s/" % i
            if not os.path.isdir(indices_path):
                indices_path = args.indices_path[:-1].replace("ssd", "hdd") + "_axis_%s/" % i
            for subject_id in os.listdir(data_path):
                if os.path.isdir(data_path + subject_id):
                    p = Process(target = image_reconstruction, args = (data_path, subject_id, output_shape, indices_path,
                                                                       "%s_axis%s/results/%s" % (args.save_path[:-1],i, subject_id), "%s/test/%s/" % (args.indices_path, subject_id), args.saliency))
                    p.start()
                    procs.append(p); cpt+=1
                    if cpt % args.process_lim == 0:
                        for p in procs:
                            p.join()
                        procs = []
                else:
                    print("No sub infered files found : ", data_path + subject_id)
        for p in procs:
            p.join()
        procs = []; cpt = 0
        for subject_id in os.listdir(args.save_path + "results/"):
            p = Process(target = combine_results, args = (args.save_path, subject_id))
            p.start()
            cpt+=1
            if cpt % args.process_lim == 0:
                for p in procs:
                    p.join()
                    procs = []
        for p in procs:
            p.join()
            
