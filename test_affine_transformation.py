from scipy.ndimage import affine_transform
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_img
from matplotlib import pyplot as plt

# x = nib.load("Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz")
# image = x.get_fdata()
# M = np.random.random((4,4))
# M[3] = [0,0,0,1]
# image_tr = affine_transform(image,M)
# new_image = nib.Nifti1Image(image_tr, affine=np.eye(4))
# new_image.to_filename("deformed.nii")
# plot_img("Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz")
# plot_img("deformed.nii")
# plt.show()

image  = np.random.random((3,3,3))*100
image = np.around(image)
M = np.eye(4)
M[0][0] = 1.5

print("TRANSFORMATION MATRIX")
print(M)
print("")

image_tr = affine_transform(image,M, mode ="nearest")
print("INPUT IMAGE")
print(image)

print("")
print("OUTPUT IMAGE")
print(image_tr)


import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image,alpha, sigma):
    shape = image.shape
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dy = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dz = np.zeros_like(dx)
    x,y,z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y + dy,(-1,1)),np.reshape(x + dx,(-1,1)), np.reshape(z,(-1,1))
    return map_coordinates(image,indices).reshape(shape)


# affine transformation
# elastic_transform