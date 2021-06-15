import imageio
import numpy as np
from skimage import measure
from skimage.morphology import remove_small_objects
from shapely.geometry import Polygon

def mask2polygon(mask):
    mask = remove_small_objects(mask==255, min_size=100, connectivity=1)
    contours = measure.find_contours(mask)
    simplified = []
    for contour in contours:
        # temp = np.array(contour, dtype=np.int32)
        # coords = np.unique(temp, axis=0)
        contour = np.vstack((contour[:, 1], contour[:, 0])).T
        polygon = Polygon(contour)
        simplified.append(np.array(polygon.simplify(0.7).exterior.coords).tolist())
        
    return simplified