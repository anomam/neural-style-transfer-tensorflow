"""Utils module to help NST workflow"""
import matplotlib.pyplot as plt
from PIL import Image as Pimage
from keras.preprocessing import image
from typing import Tuple
from imageio import imwrite
import numpy as np

# From Keras, names of PIL interpolation methods
_PIL_INTERPOLATION_METHODS = {
    'nearest': Pimage.NEAREST,
    'bilinear': Pimage.BILINEAR,
    'bicubic': Pimage.BICUBIC,
}


class Image:
    """Class the will represent an image in Python"""

    def __init__(self, content: np.ndarray, label):
        """Create image from numpy array content.

        Parameters
        ----------
        content
            Image content
        label
            Image label
        """
        self.data = content
        self.label = label

    @classmethod
    def from_fp(cls, fpath: str, label=None, target_size: Tuple = None,
                interpolation: str = 'bilinear'):
        """Load image from filepath

        Parameters
        ----------
        fpath
            Path to image
        label
            Image label (default = None)
        target_size
            Tuple of width and height to rescale image (default = None)
        interpolation
            Interpolation scheme to use for rescaling

        Returns
        -------
        Created image
        """
        pil_img = image.load_img(fpath, target_size=target_size,
                                 interpolation=interpolation)
        data = image.img_to_array(pil_img).astype('uint8')
        return cls(data, label)

    def resize(self, width: int, height: int, resample: str = 'bilinear'):
        """
        Resize image to given sizes.

        Parameters
        ----------
        width
            New width of image
        height
            New height of image
        resample
            Interpolation scheme to use for rescaling image
            (default = bilinear)
        """
        pimage = Pimage.fromarray(self.data)
        interp_method = _PIL_INTERPOLATION_METHODS[resample]
        pimage = pimage.resize((width, height), resample=interp_method)
        self.data = image.img_to_array(pimage).astype('uint8')

    def show(self, ax: plt.Axes = None, figsize: tuple = (3, 3),
             hide_axis: bool = True):
        """Show image using Matplotlib axes.

        Parameters
        ----------
        ax
            Axes to use for showing image (default = None, will be created)
        figsize
            Figure size to use (default = (3, 3))
        hide_axis
            Flag to decide to show axis or not (default = True, hide it)
        """
        ax = show_image(self, ax=ax, figsize=figsize, hide_axis=hide_axis)
        ax.set_title('label: {}'.format(self.label))

    def save(self, fpath: str):
        """Write image to path.

        Parameters
        ----------
        fpath
            Path to use to write file
        """
        imwrite(fpath, self.data)

    @property
    def as_pil(self):
        """Get image as a PIL image object"""
        return Pimage.fromarray(self.data)


def show_image(img: Image, ax: plt.Axes = None, figsize: tuple = (3, 3),
               hide_axis: bool = True) -> plt.Axes:
    """Show image using Matplotlib

    Parameters
    ----------
    img
        Image object
    ax
        Axes to use for showing image (default = None, will be created)
    figsize
        Figure size to use (default = (3, 3))
    hide_axis
        Flag to decide to show axis or not (default = True, hide it)

    Returns
    -------
    Axes with plotted image
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img.data)
    if hide_axis:
        ax.axis('off')
    return ax
