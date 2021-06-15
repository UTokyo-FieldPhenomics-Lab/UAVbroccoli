import imageio
import numpy as np
from skimage import measure
from shapely.geometry import Polygon

def mask2polygon(mask):
    contours = measure.find_contours(mask)
    simplified = []
    for contour in contours:
        temp = np.array(contour, dtype=np.int32)
        coords = np.unique(temp, axis=0)
        polygon = Polygon(coords)
        simplified.append(np.array(polygon.simplify(0.2).exterior.coords))
    return simplified