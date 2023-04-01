import imageio
import numpy as np
from skimage import measure
from skimage.morphology import remove_small_objects, remove_small_holes
from shapely.geometry import Polygon

def mask2polygon(mask):
    mask = remove_small_holes(mask==255, area_threshold=100)
    mask = remove_small_objects(mask, min_size=100, connectivity=1)
    
    h, _ = mask.shape
    contours = measure.find_contours(mask)
    simplified = []
    for contour in contours:
        # temp = np.array(contour, dtype=np.int32)
        # coords = np.unique(temp, axis=0)
        contour = np.flip(contour, axis=1)
        contour[:, 1] = h - contour[:, 1]
        polygon = Polygon(contour)
        result = np.array(polygon.simplify(0.7).exterior.coords)
        result[:, 1] = h - result[: ,1]
        simplified.append(result.tolist())
        
    return simplified