import nibabel as nib
import skimage
from skimage import exposure, segmentation, color
from skimage import feature
from skimage.future import graph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from nibabel.testing import data_path
import os
from IPython.display import *

from skimage.filters import sobel
 

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices), figsize=(8, 3), sharex=True, sharey=True)
    for i, slice in enumerate(slices):
       axes[i].imshow(slice)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                            bottom=0.02, left=0.02, right=0.98)
    plt.show()

def image_segmentation(in_file_name, out_file_name_x, out_file_name_y, out_file_name_z, show_image):
    #example_ni1 = os.path.join(data_path, in_file_name)
    n1_img = nib.load(in_file_name)
    img_data = n1_img.get_data()
    print(img_data.shape)
    #save_example_ni1 = os.path.join(data_path, out_file_name)

    slice_x = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]))
    slice_y = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]))
    slice_z = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]))
    edges_x = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]), dtype=np.int16)
    edges_y = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]), dtype=np.int16)
    edges_z= np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]), dtype = np.int16)
    for i in range(img_data.shape[0]):
        slice_x[i,:,:] = img_data[i,:,:,0]
        slice_x[i,:,:] = exposure.rescale_intensity(slice_x[i,:,:], out_range=(0, 256))
        edges_x[i,:,:] = sobel(slice_x[i,:,:])

    for i in range(img_data.shape[1]):
        slice_y[:, i, :] = img_data[:, i, :, 0]
        slice_y[:, i, :] = exposure.rescale_intensity(slice_y[:, i, :], out_range=(0, 256))
        edges_y[:, i, :] = sobel(slice_y[:, i, :])

    for i in range(img_data.shape[2]):
        slice_z[:, :, i] = img_data[:, :, i, 0]
        slice_z[:, :, i] = exposure.rescale_intensity(slice_z[:, :, i], out_range=(0, 256))
        edges_z[:, :, i] = sobel(slice_z[:, :, i])
        #slice[i] = exposure.rescale_intensity(slice[i], out_range=(0, 256))
        #img = color.gray2rgb(slice[i])

            #out1 = color.label2rgb(labels1, img, kind='avg')
            #g = graph.rag_mean_color(img, labels1, mode='similarity')
            #labels2 = graph.cut_normalized(labels1, g)
            #out2 = color.label2rgb(labels2, img, kind='avg')
            #segm[i] = color.rgb2gray(out1)
            #segm[i] = out1
            #print(segm[i].dtype)


    if (show_image):
        show_slices([slice_x[100], slice_x[110], slice_x[120]])
        plt.suptitle("slices")

    for i in range(img_data.shape[0]):
        img_data[i,:,:,0] = edges_x[i,:,:]

    save_img = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(save_img, out_file_name_x)

    for i in range(img_data.shape[1]):
        img_data[:, i, :, 0] = edges_y[:, i, :]
    save_img = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(save_img, out_file_name_y)

    for i in range(img_data.shape[2]):
        img_data[:, :, i, 0] = edges_z[:, :, i]
    save_img = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(save_img, out_file_name_z)

    if (show_image):
        # display results
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

        ax1.imshow(img_data[:,:,100,0])
        ax1.axis('off')
        ax1.set_title('image 100', fontsize=20)

        ax2.imshow(img_data[:,:,110,0])
        ax2.axis('off')
        ax2.set_title('image 110', fontsize=20)

        ax3.imshow(img_data[:,:,120,0])
        ax3.axis('off')
        ax3.set_title('image 120', fontsize=20)

        fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                            bottom=0.02, left=0.02, right=0.98)

        plt.show()


    return 0

# main begins
# load data

train_dataFile_size = 278
test_dataFile_size = 138


for i in range(0,train_dataFile_size-1):
    file_name_input = "../data/set_train/train_" + str(i+1) + ".nii"
    # example_ni1 = os.path.join(data_path, file_name_input)
    file_name_output_x = "../data/set_train_ed_x/train_" + str(i+1) + ".nii"
    file_name_output_y = "../data/set_train_ed_y/train_" + str(i + 1) + ".nii"
    file_name_output_z = "../data/set_train_ed_z/train_" + str(i + 1) + ".nii"
    # save_example_ni1 = os.path.join(data_path, file_name_output)
    image_segmentation(file_name_input,file_name_output_x,file_name_output_y,file_name_output_z, show_image=False)

for i in range(0,test_dataFile_size-1):
    file_name_input = "../data/set_test/test_" + str(i+1) + ".nii"
    # example_ni1 = os.path.join(data_path, file_name_input)
    file_name_output_x = "../data/set_test_ed_x/test_" + str(i+1) + ".nii"
    file_name_output_y = "../data/set_test_ed_y/test_" + str(i + 1) + ".nii"
    file_name_output_z = "../data/set_test_ed_z/test_" + str(i + 1) + ".nii"
    # save_example_ni1 = os.path.join(data_path, file_name_output)
    image_segmentation(file_name_input, file_name_output_x, file_name_output_y, file_name_output_z, show_image=False)
