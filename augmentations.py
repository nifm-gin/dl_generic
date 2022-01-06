import numpy as np
from scipy import signal
import random
### IMPLEMENTED AUGMENTATIONS SCRIPTS ###

def gamma(scan, g = None):
    data = np.zeros_like(scan)
    gamma = random.uniform(0.4, 1.6) if g is None else g
    try:
        for i in range(scan.shape[-1]):
            mn = np.amin(scan[:, :, :, :, i])
            rng = np.amax(scan[:, :, :, :,  i]) - mn
            data[:,:, :, :, i] = np.power((scan[:, :, :, :, i] - mn) / (rng + 1e-7), gamma)
        return data
    except :
        try :
            for i in range(scan.shape[-1]):
                mn = np.amin(scan[:, :, :, i])
                rng = np.amax(scan[:, :, :, i]) - mn
                data[:, :, :, i] = np.power((scan[:, :, :, i] - mn) / (rng + 1e-7), gamma)
            return data
        except:
            try:
                #2D patch
                for i in range(scan.shape[-1]):
                    mn = np.amin(scan[:, :, i])
                    rng = np.amax(scan[:, :, i]) - mn
                    data[:, :, i] = np.power((scan[:, :, i] - mn) / (rng + 1e-7), gamma)
                return data
            except Exception as err:
                print("Could not gamma scan, check dimensions!", err)

def blur(scan):
    data = np.zeros_like(scan)
    size = 2
    try:
        kernel = np.ones((size, size, size)) / (size**3)
        for i in range(scan.shape[-1]):
            data[:,:,:,i] = signal.convolve(scan[:, :, :, i], kernel, mode = "same")
        return data
    except:
        try:
            #2D scan / patch
            kernel = np.ones((size, size)) / (size**2)
            for i in range(scan.shape[-1]):
                data[:,:,i] = signal.convolve(scan[:, :, i], kernel, mode = "same")
            return data
        except Exception as err:
            print("Could not blur scan, check dimensions!", err)
    
def elastic_transform(scan, std_dev= 0.5):
    import SimpleITK as sitk
    interpolator = "cubic"
    sitk_image = sitk.GetImageFromArray(scan)
    try:
        transform_mesh_size = [int(i * 0.25) for i in scan.shape]
        transform = sitk.BSplineTransformInitializer(
            sitk_image ,
            transform_mesh_size
        )
        # Read the parameters as a numpy array, then add random
        # displacement and set the parameters back into the transform
        params = np.asarray(transform.GetParameters(), dtype=np.float64)
        params = params + np.random.randn(params.shape[0]) * std_dev
        transform.SetParameters(tuple(params))
        # Create resampler object
        # The interpolator can be set to sitk.sitkBSpline for cubic interpolation,
        # see https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5 for more options
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_image)
        resampler.SetInterpolator(sitk.sitkLinear if interpolator == 'linear' else sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(float(scan.min())) # Fill with minimu value if out of image, float bcause otherwise tf thinks it's a double
        resampler.SetTransform(transform)
        # Execute augmentation
        sitk_augmented_image = resampler.Execute(sitk_image)
        # Convert back to numpy
        augmented_image = sitk.GetArrayFromImage(sitk_augmented_image)
        data = augmented_image.astype(dtype=np.float32)
        data[data < -1] = -1
        data[data > 1] = 1
        data = data.reshape(scan.shape)
        return data
    except Exception as err:
        print("Could not process elastic transform, check input dimension", err)

def flip_x(scan):
    #return scan flipped on x axis
    return np.flip(scan, axis = 0)
def flip_y(scan):
    #return scan flipped on x axis
    return np.flip(scan, axis = 1)
def flip_z(scan):
    #return scan flipped on x axis
    return np.flip(scan, axis = 2)
def translate_x(scan):
    copy = np.copy(scan)
    shape = list(copy.shape)
    shape[0] += 2
    copy.resize(shape)
    copy[-2:] = np.amin(copy)
    return copy[2:]
def translate_y(scan):
    copy = np.copy(scan)
    shape = list(copy.shape)
    shape[1] += 2
    copy.resize(shape)
    copy[:,-2:] = np.amin(copy)
    return copy[:,2:]
    
def translate_z(scan):
    try:
        copy = np.copy(scan)
        shape = list(copy.shape)
        shape[2] += 2
        copy.resize(shape)
        copy[:, :, -2:] = np.amin(copy)
        return copy[:, :, 2:]
    except:
        print("Could not translate scan on z axis, check dimension")
#Dict ref name : fct
dict_augmentations = {
    "flip_x" : flip_x,
    "flip_y" : flip_y,
    "flip_z" : flip_z,
    "translate_x" : translate_x,
    "translate_y" : translate_y,
    "translate_z" : translate_z,
    "elastic" : elastic_transform,
    "blur" : blur,
    "gamma" : gamma
    }
 
