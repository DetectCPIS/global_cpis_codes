from osgeo import gdal
import numpy as np
import os
import copy

def get_image_info(fileName):

    dataset = gdal.Open(fileName)
    if dataset == None:
        print("Can't Open" + fileName)
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    return im_width, im_height, im_bands, im_geotrans, im_proj


def readTiff(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print("Can't Open" + fileName)
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    return im_width, im_height, im_data, im_bands, im_geotrans, im_proj



