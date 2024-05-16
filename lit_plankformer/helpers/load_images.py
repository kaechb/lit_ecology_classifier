import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ..helpers.image_edits import ResizeWithoutProportions, ResizeWithProportions


def load_image(filename, L=None, resize_images=None, show=False):
    """ Loads one image, and rescales it to size L.
    The pixel values are between 0 and 255, instead of between 0 and 1, so they should be normalized outside of the function
    """

    image = Image.open(filename)
    # Set image's largest dimension to target size, and fill the rest with black pixels
    if not resize_images:
        rescaled = 0
    elif resize_images == 1:
          # width and height are assumed to be the same (assertion at the beginning)
        image, rescaled = ResizeWithProportions(image,L)
    elif resize_images == 2:
         # width and height are assumed to be the same (assertion at the beginning)
        image, rescaled = ResizeWithoutProportions(image,L)
    npimage = np.array(image.copy(), dtype=np.float32)
    if show:
        plt.figure()
        os.makedirs("debug",exist_ok=True)
        plt.imshow(np.array(image))
        plt.savefig("debug/test.png")
        plt.close()
    image.close()
    return npimage, rescaled



def load_images(datapaths, L, resize_images=None):
    """
    Loads images from specified directories, processes them, and returns a DataFrame containing
    the image data and filenames.

    Arguments:
    datapaths     - List of base paths where the image data is stored.
    L             - Target size to which images are rescaled (LxL, maintaining proportions).
    resize_images - If provided, specifies additional resizing options.


    Returns:
    df            - DataFrame with columns ['filename', 'npimage'] containing image data.
    """
    # Define pattern based on directory structure.
    file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    # Generate list of image paths.
    image_paths = []
    for base_path in datapaths:
        for pattern in file_patterns:
            image_paths.extend(glob.glob(os.path.join(base_path, pattern)))
    # Load images and create DataFrame entries.
    data = []
    for image_path in image_paths:
        npimage, _ = load_image(image_path, L, resize_images,)
        data.append({'filename': image_path, 'npimage': npimage / 255.0})
    # Create and DO NOT shuffle DataFrame.
    df = pd.DataFrame(data)
    return df