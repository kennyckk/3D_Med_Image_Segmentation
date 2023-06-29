import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask, folder='./predicted_mask'):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1,figsize=(20,10))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.savefig(folder + '/class_map.jpg')
    plt.show()

def show_slices(slices):
    """
    Function to display a row of image slices
    Input is a list of numpy 2D image slices
    """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

    plt.show()

def show_mid_slice(img_numpy, title='img'):
   """
   Accepts an 3D numpy array and shows median slices in all three planes
   """
   assert img_numpy.ndim == 3
   n_i, n_j, n_k = img_numpy.shape

   # sagittal (left image)
   center_i1 = int((n_i - 1) / 2)
   # coronal (center image)
   center_j1 = int((n_j - 1) / 2)
   # axial slice (right image)
   center_k1 = int((n_k - 1) / 2)

   show_slices([img_numpy[center_i1, :, :],
                img_numpy[:, center_j1, :],
                img_numpy[:, :, center_k1]])
   plt.suptitle(title)