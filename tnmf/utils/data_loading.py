# pylint: disable=missing-module-docstring
import numpy as np

from scipy.misc import face
from PIL import Image


def racoon_image(gray: bool = True, scale: float = 1.):
    img = Image.fromarray(face(gray))
    img = img.resize([int(scale * s) for s in img.size])
    img = np.array(img, dtype=float) / 255
    return img
