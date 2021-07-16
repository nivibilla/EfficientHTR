from . import pageBordering
from . import lineRemoval
from . import circleRemoval
import cv2
import numpy as np

def processImage(img, iterations=3):
    """Cleans the image by bordering it and removing any page holes/lines."""
    # Border the image to page
    error, bordered = pageBordering.page_border(img.copy())
    if error:
        raise Exception("The image provided could not be bordered.")

    # Removes page holes
    holes_removed = circleRemoval.page_hole_removal(bordered)

    # Remove lines on lined paper (repeating for multiple iterations gives better results)
    lines_removed = holes_removed
    for i in range(iterations):
        lines_removed, gray = lineRemoval.lines_removal(lines_removed)

    return preprocessImage(lines_removed)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=2.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def preprocessImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0])
    upper = np.array([179, 255, 209])
    mask = cv2.inRange(image, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    image[close==0] = (255,255,255)
    retouch_mask = (image <= [250.,250.,250.]).all(axis=2)
    image[retouch_mask] = [0,0,0]
    image = 255-image
    return image[...,0]