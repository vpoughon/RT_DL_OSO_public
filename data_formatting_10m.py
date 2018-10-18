# -*- coding: utf-8 -*-

'''
R&T generation de cartes d'occupation des sols par reseaux de neurones convolutionnels

Data formatting in order to generate inputs for convolutionnal neural network

Requirement: input images and vector data must be in the same projection (use of Lambert93 over France)
Inputs:
    - Directory containing Sentinel2 images
    - Vector files (shp) containing labeled data (training and testing)
    - Output directory that will contain formatted images

Processing:
    - Rasterize label vector data in input image geometry

Outputs:
    - train and test raster images 

Execution example:
python data_formatting_10m.py -img /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016/ -label /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016/ReferenceData_by_tile -out /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016/RasterData

Author : Vincent Poulain (2017), Thales Services
'''

import os
import argparse
import glob
import sys
import numpy as np
import subprocess
import gdal
import gdalconst
from osgeo import ogr
import osgeo.gdal_array

def get_image_list(s_img_dir):
    '''
    get the list of images in the input directory
    @param s_img_dir: input directory
    @return t_img: list of input images
    '''
    t_img = glob.glob(os.path.join(s_img_dir, 'T?????/*.tif'))
    if len(t_img) == 0:
        raise Exception('Error, no image found in {}'.format(s_img_dir))
    return t_img

def raster_to_2D_array(s_img, u_band):
    '''
    create a 2D numpy array from a raster image. Output array shape is : (nb_row, nb_col)
    @param s_img: multi-band input image
    @param u_band: number of band to extract (starting from 1)
    @return t_img: numpy array corresponding to the input image, shape : (nb_row, nb_col)
    @return o_proj: projection information of the input image
    @return o_geo: geometric information of the input image
    '''
    o_image = gdal.Open(s_img)
    if o_image is None:
        print("couldn't open input dataset {0}".format(s_img))
        sys.exit(1)
    o_proj = o_image.GetProjection()
    if 'LOCAL_CS' in o_proj and not 'PROJCS' in o_proj:
        # bug with a gdal version that prevent API to get projection PROJCS (replaced by LOCAL_CS)
        o_proj = read_projection_gdalinfo(s_img)
    o_geo = o_image.GetGeoTransform()
    u_nb_col = o_image.RasterXSize
    u_nb_row = o_image.RasterYSize
    o_band = o_image.GetRasterBand(u_band)
    t_img = np.array(o_band.ReadAsArray(0, 0, u_nb_col, u_nb_row))
    o_image = None
    return t_img, o_proj, o_geo

def raster_to_3D_array(s_img):
    '''
    create a 3D numpy array from a raster image. Output array shape is : (nb_band, nb_row, nb_col)
    @param s_img: multi-band input image
    @return t_img: numpy array corresponding to the input image, shape : (nb_band, nb_row, nb_col)
    @return o_proj: projection information of the input image
    @return o_geo: geometric information of the input image
    '''
    o_image = gdal.Open(s_img)
    if o_image is None:
        print("couldn't open input dataset {0}".format(s_img))
        sys.exit(1)
    o_proj = o_image.GetProjection()
    if 'LOCAL_CS' in o_proj and not 'PROJCS' in o_proj:
        # bug with a gdal version that prevent API to get projection PROJCS (replaced by LOCAL_CS)
        o_proj = read_projection_gdalinfo(s_img)
    o_geo = o_image.GetGeoTransform()
    t_img = o_image.ReadAsArray()
    o_image = None
    return t_img, o_proj, o_geo

def img_list_to_4D_array(t_img_list):
    '''
    create a 4D array from a list of images
    @param t_img_list : list of images
    @return t_4D_array : 4D numpy array. Shape: (nb of images, nb of bands, nb of rows, nb of cols)
    '''
    t_4D_array = []
    for s_img in t_img_list:
        t_img, _, _ = raster_to_3D_array(s_img)
        t_4D_array.append(t_img)
    return np.array(t_4D_array)

def array_3D_to_raster(t_img, s_out_img, o_proj=None, o_geo=None, b_aux=1):
    '''
    write an image from a 3D numpy array
    @param t_img: numpy array, shape : (nb_band, nb_row, nb_col)
    @param s_out_img: filename to write
    @param o_proj: projection information of the output image
    @param o_geo: geometric information of the output image
    @param b_aux: boolean to activate the writing of projection and geo information in a .aux.xml file
    '''
    #o_out = osgeo.gdal_array.SaveArray(np.int16(t_img), s_out_img)
    o_out = osgeo.gdal_array.SaveArray(t_img, s_out_img)
    o_out = None
    # add geolocation information
    o_ds_out = gdal.Open(s_out_img, gdalconst.GA_Update)
    if o_proj != None:
        o_ds_out.SetProjection(o_proj)
    if o_geo != None:
        o_ds_out.SetGeoTransform(o_geo)
    o_ds_out = None 
    # bug with a gdal version that prevent API to write projection in GEOTIFF files
    if b_aux and o_proj != None and o_geo != None:
        s_aux = s_out_img + '.aux.xml'
        o_aux = open(s_aux, 'w')
        o_aux.write('<PAMDataset>\n')
        o_aux.write('<SRS>{0}</SRS>\n'.format(o_proj))
        o_aux.write('<GeoTransform>{0}</GeoTransform>\n'.format(str(list(o_geo)).replace('[', '').replace(']', '')))
        o_aux.write('</PAMDataset>\n')
        o_aux.close()
    return 0

def raster_add_bands(s_img_in, s_img_out, t_bands):
    '''
    add bands to an existing array and fill them with the input 3D numpy array
    @param s_img_in: filename of the input image
    @param s_img_out: filename of the output image
    @param t_bands: 3D numpy array of the bands to add
    '''
    t_img, o_proj, o_geo = raster_to_3D_array(s_img_in)
    t_full = np.concatenate((t_img, t_bands), axis=0)
    array_3D_to_raster(t_full, s_img_out, o_proj, o_geo)
    return 0

def rasterize_labels(t_input_img, s_label, s_out):
    '''
    rasterize vector labels in input image geometry
    @param t_input_img: list of input images
    @param s_label: label directory name
    @param s_out: output directory
    '''
    os.system('mkdir -p {0}'.format(s_out))
    # for each input image:
    for s_input_img in t_input_img:
        o_driver = ogr.GetDriverByName('ESRI Shapefile')
        s_tile = os.path.basename(os.path.dirname(s_input_img))
        s_label_training = os.path.join(s_label, s_tile, 'training.shp')
        s_label_testing = os.path.join(s_label, s_tile, 'testing.shp')
        for s_vector_label in [s_label_training, s_label_testing]:
            s_case = os.path.splitext(os.path.basename(s_vector_label))[0]
            o_dataset = o_driver.Open(s_vector_label)
            o_layer = o_dataset.GetLayer()
            s_out_img = os.path.join(s_out, s_tile + '_label_{0}.tif'.format(s_case))
            # read geo et proj info of input image
            t_img, o_proj, o_geo = raster_to_2D_array(s_input_img, 1)
            # write a black image
            #print t_img.shape
            t_img_out = np.zeros((t_img.shape[0], t_img.shape[1]), dtype=np.byte) #(nb_row, nb_col)
            array_3D_to_raster(t_img_out, s_out_img, o_proj, o_geo)
            o_img = gdal.Open(s_out_img, gdalconst.GA_Update)
            # rasterize according to "CODE" value, representing the class of the label
            gdal.RasterizeLayer(o_img, [1], o_layer, burn_values=[0], options=["ATTRIBUTE=CODE2"])

def read_projection_gdalinfo(s_img_src):
    '''
    read projection information of input image
    developped because of a bug in Gdal API : GetProjection() can return LOCAL_CS instead of PROJCS
    As gdalinfo return right projection information, it is used to get the Projection
    @param s_img_src : source image from which projection information is read
    @return s_proj : projection information
    '''
    p = subprocess.Popen(["gdalinfo "+s_img_src], shell=True, stdout=subprocess.PIPE)
    value = p.communicate()[0].decode("utf-8")
    if 'Coordinate System is:\n' in value and 'Origin =' in value:
        s_proj = value.split('Coordinate System is:\n')[1].split('Origin =')[0]
    else:
        print('Warning, failure in projection retrieval using gdalinfo')
        s_proj = ''
    return s_proj


def main():
    '''
    main
    '''
    # Parser creation
    parser = argparse.ArgumentParser(description='Sentinel2 data formatting for deep learning')
    # Args
    parser.add_argument('-img', '--img', metavar='[IMG_DIR]', help='Sentinel2 image directory', required=True)
    parser.add_argument('-label', '--label', metavar='[LABEL]',\
                        help='Directory containing for each tile training.shp and testing.shp', required=True)
    parser.add_argument('-out', '--out', metavar='[OUT]', help='Output directory', required=True)
    # Command line parsing
    args = vars(parser.parse_args())
    s_img_dir = os.path.abspath(args['img'])
    s_label = os.path.abspath(args['label'])
    s_out = os.path.abspath(args['out'])
    os.system('mkdir -p {0}'.format(s_out))
    # get input images:
    t_img = get_image_list(s_img_dir)
    # Rasterize labels:
    rasterize_labels(t_img, s_label, s_out)
    

if __name__ == '__main__':
    main()
