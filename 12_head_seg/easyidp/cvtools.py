import numpy as np
from skimage.draw import polygon2mask
from shapely.geometry import MultiPoint, Polygon

# ignore the warning of shapely convert coordiante
import warnings
warnings.filterwarnings("ignore", message="The array interface is deprecated and will no longer work in Shapely 2.0")


def imarray_crop(imarray, polygon_hv, outside_value=0):
    """crop a given ndarray image by given polygon pixel positions

    Parameters
    ----------
    imarray : ndarray
        | the image data in numpy ndarray
        | if the shape is (height, width), view it as DSM data, the data type should be float.
        | if the shape is (height, width, dimen), view it as RGB DOM data (dimen=3 means RGB and dimen=4 means RGBA).
        |     the data type for this case should be either 0-1 float, or 0-255 int.

        .. caution::

            Currently, the EasyIDP package does not have the ability to handle multi-spectral image data directly.
            If you really want to use this function to crop multi-spectral image with multiple layers, please send each layer one by one.

            For example, you have a multi-spectral imarray with 6 bands:

            .. code-block:: python

                >>> multi_spect_imarray.shape
                (1028, 800, 6)

            Then using the following for loops to iteratively process each band 
            (please modify it by yourself, can not guarantee it works directly)

            .. code-block:: python

                >>> band_container = []
                >>> for i in range(0, 6):
                >>>     band = multi_spect_imarray[:,:,i]
                >>>     out, offset = idp.cvtools.imarray_crop(band, polygon_hv, outside_value=your_geotiff.header['nodata'])
                >>>     band_container.append(out)
                >>> final_out = np.dstack(band_container)

    polygon_hv : 2D ndarray
        | pixel position of boundary point, the order is (horizontal, vertical)
        
        .. caution::

            it is reverted to the numpy imarray axis. 
            horzontal = numpy axis 1, vertical = numpy axis 0.

    outside_value: int | float
        | specify exact value outside the polgyon, default 0.
        | But for some DSM geotiff, it could be -10000.0, depends on the geotiff meta infomation

    returns
    -------
    imarray_out : ndarray
        the (m,n,d) ndrray to store pixel info
    roi_top_left_offset : ndarray
        the (h, v) pixel index that represent the polygon bbox left top corner

    """
    # check if the imarray is correct imarray
    if not isinstance(imarray, np.ndarray) or \
        not (
            np.issubdtype(imarray.dtype, np.integer) \
                or \
            np.issubdtype(imarray.dtype, np.floating)
            ):
        raise TypeError(f"The `imarray` only accept numpy ndarray integer and float types")


    # check if the polygon_hv is float or int, or in proper shape
    # fix the roi is float cause indexing error: github issue #61
    if not isinstance(polygon_hv, np.ndarray):
        raise TypeError(f"Only the numpy 2d array is accepted, not {type(polygon_hv)}")
    if len(polygon_hv.shape) != 2 or polygon_hv.shape[1] !=2:
        raise AttributeError(f"Only the 2d array (xy) is accepted, expected shape like (n, 2),  not current {polygon_hv.shape}")

    if np.issubdtype(polygon_hv.dtype, np.integer):
        pass
    elif np.issubdtype(polygon_hv.dtype, np.floating):
        polygon_hv = polygon_hv.astype(np.uint32)
    else:
        raise TypeError(f"Only polygon coordinates with [np.interger] and [np.floating]"
                        f" are accepted, not dtype('{polygon_hv.dtype}')")

    # (horizontal, vertical) remember to revert in all the following codes
    roi_top_left_offset = polygon_hv.min(axis=0)
    roi_max = polygon_hv.max(axis=0)
    roi_length = roi_max - roi_top_left_offset

    roi_rm_offset = polygon_hv - roi_top_left_offset
    # the polygon will generate index outside the image
    # this will cause out of index error in the `poly2mask`
    # so need to find out the point locates on the maximum edge and minus 1 
    # >>> a = np.array([217, 468])  # roi_max
    # >>> b  # polygon
    # array([[217, 456],
    #        [ 30, 468],
    #        [  0,  12],
    #        [187,   0],
    #        [217, 456]])
    # >>> b[:,0] == a[0]
    # array([ True, False, False, False,  True])
    # >>> b[b[:,0] == a[0], 0] -= 1
    # >>> b
    # array([[216, 456],
    #        [ 30, 468],
    #        [  0,  12],
    #        [187,   0],
    #        [216, 456]])
    roi_rm_offset[roi_rm_offset[:,0] == roi_length[0], 0] -= 1
    roi_rm_offset[roi_rm_offset[:,1] == roi_length[1], 1] -= 1

    # remove (160, 160, 1) such fake 3 dimention
    imarray = np.squeeze(imarray)
    
    dim = len(imarray.shape)
    if dim == 2: 
        # only has 2 dimensions
        # e.g. DSM 1 band only, other value outside polygon = empty value
        
        # here need to reverse 
        # imarray.shape -> (h, w), but poly2mask need <- (w, h)
        roi_cropped = imarray[roi_top_left_offset[1]:roi_max[1], 
                              roi_top_left_offset[0]:roi_max[0]]
        rh = roi_cropped.shape[0]
        rw = roi_cropped.shape[1]
        mask = poly2mask((rw, rh), roi_rm_offset)

        roi_cropped[~mask] = outside_value
        imarray_out = roi_cropped

    elif dim == 3: 
        # has 3 dimensions
        # e.g. DOM with RGB or RGBA band, other value outside changed alpha layer to 0
        # coordinate xy reverted between easyidp and numpy
        roi_cropped = imarray[roi_top_left_offset[1]:roi_max[1], 
                              roi_top_left_offset[0]:roi_max[0]]

        rh = roi_cropped.shape[0]
        rw = roi_cropped.shape[1]
        layer_num = roi_cropped.shape[2]

        # here need to reverse 
        # imarray.shape -> (h, w), but poly2mask need <- (w, h)
        mask = poly2mask((rw, rh), roi_rm_offset)

        if layer_num == 3:  
            # DOM without alpha layer - RGB
            # but easyidp will add masked alpha layer to output.

            # change mask data type to fit with the image data type
            if np.issubdtype(roi_cropped.dtype, np.integer) and roi_cropped.min() >= 0 and roi_cropped.max() <= 255:
                # the image is 0-255 & int type
                roi_cropped = roi_cropped.astype(np.uint8)
                mask = mask.astype(np.uint8) * 255
            elif np.issubdtype(roi_cropped.dtype, np.floating) and roi_cropped.min() >= 0 and roi_cropped.max() <= 1:
                # the image is 0-1 & float type
                mask = mask.astype(roi_cropped.dtype)
            else:
                raise AttributeError(f"Can not handle RGB imarray ranges ({roi_cropped.min()} - {roi_cropped.max()}) with dtype='{roi_cropped.dtype}', "
                                    f"expected (0-1) with dtype='float' or (0-255) with dtype='int'")

            # merge alpha mask with cropped images
            imarray_out = np.concatenate([roi_cropped, mask[:, :, None]], axis=2)

        elif layer_num == 4:  
            # DOM with alpha layer - RGBA

            # merge orginal mask with polygon_hv mask
            original_mask = roi_cropped[:, :, 3].copy()
            original_mask = original_mask > 0    # change type to bool
            merged_mask = original_mask * mask   # bool = bool * bool

            # change mask data type to fit with the image data type
            if np.issubdtype(roi_cropped.dtype, np.integer) and roi_cropped.min() >= 0 and roi_cropped.max() <= 255:
                # the image is 0-255 & int type
                roi_cropped = roi_cropped.astype(np.uint8)
                merged_mask = merged_mask.astype(np.uint8) * 255
            elif np.issubdtype(roi_cropped.dtype, np.floating) and roi_cropped.min() >= 0 and roi_cropped.max() <= 1:
                # the image is 0-1 & float type
                merged_mask = merged_mask.astype(roi_cropped.dtype)
            else:
                raise AttributeError(f"Can not handle RGB imarray ranges ({roi_cropped.min()} - {roi_cropped.max()}) with dtype='{roi_cropped.dtype}', "
                                    f"expected (0-1) with dtype='float' or (0-255) with dtype='int'")

            imarray_out = np.dstack([roi_cropped[:,:, 0:3], merged_mask])
        else:
            raise TypeError(f'Unable to solve the layer/band number {layer_num}, only one band DSM or 3|4 band RGB|RGBA DOM are acceptable')
    else:
        raise ValueError(
            f"Only image dimention=2 (mxn) or 3(mxnxd) are accepted, not current"
            f"[shape={imarray.shape} dim={dim}], please check whether your ROI "
            f"is smaller than one pixel.")

    return imarray_out, roi_top_left_offset


def poly2mask(image_shape, poly_coord, engine="skimage"):
    """convert vector polygon to raster masks

    Parameters
    ----------
    image_shape : tuple with 2 element
        .. caution::
            it is reversed with numpy index order 

        (horizontal, vertical) = (width, height)

    poly_coord : (n, 2) np.ndarray -> dtype = int or float
        .. caution::
            The xy is reversed with numpy index order

            (horizontal, vertical) = (width, height)
            
    engine : str, default "skimage"
        | "skimage" or "shapely"; the "pillow" has been deprecated;
        | skimage - ``skimage.draw.polygon2mask``, the default method;
        | pillow is slight different than "skimage", deprecated;
        | shapely is almost the same with "skiamge", but effiency is very slow, not recommended.

    Returns
    -------
    mask : numpy.ndarray
        the generated binary mask
        
    Notes
    -----
    This code is inspired from [1]_ .

    And for the poly_coord, if using **shapely** engine, it will following this logic for int and float:

    If dtype is int -> view coord as pixel index number
        Will + 0.5 to coords (pixel center) as judge point
    if dtype is float -> view coords as real coord
        (0,0) will be the left upper corner of pixel square

    References
    ----------
    .. [1] https://stackoverflow.com/questions/62280398/checking-if-a-point-is-contained-in-a-polygon-multipolygon-for-many-points

    """

    # check the type of input
    # is ndarray -> is int or float ndarray
    if not isinstance(poly_coord, np.ndarray) or \
        not (
            np.issubdtype(poly_coord.dtype, np.integer) \
                or \
            np.issubdtype(poly_coord.dtype, np.floating)
            ):
        raise TypeError(f"The `poly_coord` only accept numpy ndarray integer and float types")

    if len(poly_coord.shape) != 2 or poly_coord.shape[1] != 2:
        raise AttributeError(f"Only nx2 ndarray are accepted, not {poly_coord.shape}")

    w, h = image_shape

    # check whether the poly_coords out of mask boundary
    xmin, ymin = poly_coord.min(axis=0)
    xmax, ymax = poly_coord.max(axis=0)

    if engine == "shapely" and max(xmax-xmin, ymax-ymin) > 100:
        warnings.warn("Shaply Engine can not handle size over 100 efficiently, convert using pillow engine")
        engine = "skimage"

    if xmin < 0 or ymin < 0 or xmax >= w or ymax >= h:
        raise ValueError(f"The polygon coords ({xmin}, {ymin}, {xmax}, {ymax}) is out of mask boundary [0, 0, {w}, {h}]")

    if engine == "shapely":
        mask = _shapely_poly2mask(h, w, poly_coord)
    else:   # using pillow -> skimage
        # mask = _pillow_poly2mask(h, w, poly_coord)
        # the coordinate of xy is reversed with skimage
        mask = polygon2mask((w, h), poly_coord).T

    return mask


def _shapely_poly2mask(h, w, poly_coord):
    mask = np.zeros((h, w), dtype=bool)

    # use the pixel center as judgement points
    x = np.arange(0, w) + 0.5
    y = np.arange(0, h) + 0.5

    xx, yy = np.meshgrid(x, y)

    # get the coordinates of all pixel points
    # it is reversed with numpy index order -> [vertical, horizontal]
    pts = np.array([yy.ravel(), xx.ravel()]).T
    points = MultiPoint(pts)

    # judge the type of polygon coordinates
    if np.issubdtype(poly_coord.dtype, np.integer):
        # is int type, mainly means it represent
        # the id of int rather than coords xy values
        # -> shift 0.5 as the pixel center
        poly = Polygon(poly_coord + 0.5)
    elif np.issubdtype(poly_coord.dtype, np.floating):
        poly = Polygon(poly_coord)

    points_in = points.intersection(poly)

    # here will raise warning when obtain coords from shapely multipoints
    # -0.5 turns points center coords to point id
    # here are point index of "masked" pixels
    idx = (np.array(points_in) - 0.5).astype(int)

    # turn to masks
    # idx -> (pixel horizontal, pixel vertical)
    # it is reversed with numpy index order -> [vertical, horizontal]
    mask[idx[:,1], idx[:,0]] = True

    return mask

# def _pillow_poly2mask(h, w, poly_coord):
#     # deprecated
#     mask = Image.new('1', (w, h), color=0)
#     draw = ImageDraw.Draw(mask)

#     xy_pil = [tuple(i) for i in poly_coord]
    
#     draw.polygon(xy_pil, fill=1, outline=1)

#     mask = np.array(mask, dtype=bool)

#     return mask


def rgb2gray(rgb):
    """Transform the RGB image to gray image

    Parameters
    ----------
    rgb : mxnx3 ndarray
        The RGB ndarray image need to be converted

    Returns
    -------
    gray : mxn ndarray
        The output 2D ndarray after convension

    Notes
    -----
    Using the same formular that matplotlib did [1]_ for the transformation.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])