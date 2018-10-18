# -*- coding: utf-8 -*-

'''
R&T generation de cartes d'occupation des sols par reseaux de neurones convolutionnels

Split labeled data into train and test datasets


Inputs:
    - Directory containing Sentinel2 tiles (to know where to produce train/test data)
    - Vector files (shp) containing labeled data
    - Output directory that will contain train and test vector files

Processing:
    - Read geographic footprints of S2 images
    - Extract vector data on these footprints
    - Create train and test vector files (a polygon will not be splitted into train and test, the surface of labeled data and the number of polygon will respect the proportion of 67% for train and 33% for test)

Outputs:
    - a directory for each tile, containing 2 shp files representing train and test vector data

Execution example:
python create_train_test_label.py -img /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016/ \
-label /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016/ReferenceData/2016/echantillons_OSO_2014.shp \
-out /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016/ReferenceData_by_tile

python create_train_test_label.py -img /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ \
-label /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016/ReferenceData/2016/echantillons_OSO_2014.shp \
-out /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_by_tile

python create_train_test_label.py -img /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract \
-label /work/OT/siaa/Work/RTDLOSO/partage/RPG2016_UA/learn_CP2014_RPG2016_UA.shp \
-out /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile

Author : Vincent Poulain (2017), Thales Services
'''

import os
import argparse
import glob
import sys
import numpy as np
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

def read_tile_footprint(s_img):
    '''
    
    @return t_footprint : list of tuples (id tile, o_footprint) with o_footprint ogr geometry
    '''
    o_image = gdal.Open(s_img)
    if o_image is None:
        print("couldn't open input dataset" + s_img)
        sys.exit(1)
    o_proj = o_image.GetProjection()
    o_geo = o_image.GetGeoTransform() # (243367.3657136865, 10.0, 0.0, 6767408.600451828, 0.0, -10.0)
    u_nb_col = o_image.RasterXSize
    u_nb_row = o_image.RasterYSize
    o_image = None
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(o_geo[0], o_geo[3])
    ring.AddPoint(o_geo[0] + u_nb_col * o_geo[1], o_geo[3])
    ring.AddPoint(o_geo[0] + u_nb_col * o_geo[1], o_geo[3] + u_nb_row * o_geo[5])
    ring.AddPoint(o_geo[0], o_geo[3] + u_nb_row * o_geo[5])
    ring.AddPoint(o_geo[0], o_geo[3])
    o_poly = ogr.Geometry(ogr.wkbPolygon)
    o_poly.AddGeometry(ring)
    return (os.path.basename(os.path.dirname(s_img)), o_poly)
    
    
def label_data_by_tile(s_label, tile_footprint):
    '''
    @param tile_footprint : list of tuples (id tile, o_footprint) with o_footprint ogr geometry
    @return (tile_footprint[0], geomcol, class) : tuple containing id tile, a geometry collection composed of polygons intersecting the tile, the class value
    '''
    o_driver = ogr.GetDriverByName('ESRI Shapefile')
    o_dataset = o_driver.Open(s_label)
    o_layer = o_dataset.GetLayer()
    o_srs = o_layer.GetSpatialRef()
    # spatial filter to get only feature intersecting the tile
    o_layer.SetSpatialFilter(tile_footprint[1])
    # Collect all Geometry
    geomcol = ogr.Geometry(ogr.wkbGeometryCollection)
    t_class = []
    t_ID = []
    t_AREA_HA = []
    for feature in o_layer:
        #print(feature.GetFieldCount())
        #print(feature.GetDefnRef().GetFieldDefn(0).GetName()) # ID
        #print(feature.GetDefnRef().GetFieldDefn(1).GetName()) # AREA_HA (Area in RPG2016_UA)
        #print(feature.GetDefnRef().GetFieldDefn(2).GetName()) # CODE2 (Code in RPG2016_UA)
        s_code_name = feature.GetDefnRef().GetFieldDefn(2).GetName()
        s_area_name = feature.GetDefnRef().GetFieldDefn(1).GetName()
        geomcol.AddGeometry(feature.GetGeometryRef())
        t_class.append(feature.GetFieldAsString(s_code_name))
        t_ID.append(feature.GetFieldAsString('ID'))
        t_AREA_HA.append(float(feature.GetFieldAsString(s_area_name)))
    return (tile_footprint[0], geomcol, t_class, t_ID, t_AREA_HA), o_srs, o_layer.GetName()


def export_vector(filename, t_geometry, d_fields, driver="ESRI Shapefile", layer="layer", srs=None):
    '''
    write vector data (GML, KML, ESRI Shapefile, ...) from a list of ogr geometry
    @param filename : name of output file
    @param t_geometry : list of ogr geometry
    @param d_fields : dictionnary whose keys are features names and values are feature values
    @param driver : driver name ("GML", "KML", "ESRI Shapefile", ...)
    @param layer : layer name
    @param srs : spatial reference system of the vector data
    '''
    #       Save extent to a new GML file
    outDriver = ogr.GetDriverByName(driver)
    #       Create the output shapefile
    outDataSource = outDriver.CreateDataSource(filename)
    outLayer = outDataSource.CreateLayer(layer, srs=srs , geom_type=ogr.wkbPolygon)
    #       Add fields
    for name, val in d_fields.items():
        Field = ogr.FieldDefn(name, ogr.OFTInteger)
        outLayer.CreateField(Field)
    #       Create the feature and set values
    featureDefn = outLayer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    for i in range(len(t_geometry)): 
        feature.SetGeometry(t_geometry[i])
        for name, val in d_fields.items():
            feature.SetField(name, val[i])
        outLayer.CreateFeature(feature)
    #       Close DataSource
    outDataSource.Destroy()
    return 0

def split_label(tile_polygons, o_srs, s_layer_name, s_out):
    '''
    split labeled data between train and test
    @param tile_polygons : tuple containing (id tile, geomcol, t_class, t_ID, t_AREA_HA)
    @param o_srs : spatial reference system of vector data to write
    @param s_layer_name : layer name to write
    @param s_out : output dir
    '''
    s_tile_id = tile_polygons[0]
    s_out_tile = os.path.join(s_out, s_tile_id)
    os.system('mkdir -p {0}'.format(s_out_tile))
    o_polygons = tile_polygons[1]
    t_class = tile_polygons[2]
    t_ID = tile_polygons[3]
    t_AREA_HA = tile_polygons[4]
    t_polygons = [o_polygons.GetGeometryRef(i) for i in range(o_polygons.GetGeometryCount())]
    t_surface = [poly.GetArea() / (100.0 * 100.0) for poly in t_polygons] # en hectares
    print(s_tile_id)
    t_data = np.transpose(np.array([t_polygons, [int(classe) for classe in t_class], t_surface, t_ID, t_AREA_HA])) # each row : polygon, class, surface
    #print('labeled surface (ha) : {0}'.format(np.sum(t_data[:,2])))
    # sort by class and then surface
    t_data_sorted = np.array(sorted(t_data, key=lambda x: (x[1], x[2])))
    # for each class, on take 1/3 polygon for test, and 2/3 for train (we should get ratios of 67% / 33% for both surface and number of polygons)
    t_out = [] # output info table
    t_test_tile = np.zeros((1,5))
    t_train_tile = np.zeros((1,5))
    for u_class in sorted(set(t_data[:, 1])):
        print(u_class)
        t_data_class = t_data_sorted[np.where(t_data_sorted[:, 1] == u_class)]
        t_data_test = t_data_class[1::3, :]
        t_data_train = np.concatenate((t_data_class[0::3, :], t_data_class[2::3, :]))
        t_test_tile = np.concatenate((t_test_tile, t_data_test))
        t_train_tile = np.concatenate((t_train_tile, t_data_train))
        f_surface_class = np.sum(t_data_class[:,2])
        f_surface_train = np.sum(t_data_train[:,2])
        f_surface_test = np.sum(t_data_test[:,2])
        t_out.append([u_class, t_data_class.shape[0], f_surface_class, t_data_train.shape[0], t_data_test.shape[0], \
                      t_data_train.shape[0] * 100.0 / t_data_class.shape[0], t_data_test.shape[0] * 100.0 / t_data_class.shape[0], \
                      f_surface_train, f_surface_test, f_surface_train * 100.0 / f_surface_class, f_surface_test * 100.0 / f_surface_class])
    t_test_tile = t_test_tile[1::]
    t_train_tile = t_train_tile[1::]
    # last line : all labels:
    f_surface_tile_label = np.sum(t_data[:,2])
    f_surface_train = np.sum(t_train_tile[:,2])
    f_surface_test = np.sum(t_test_tile[:,2])
    t_out.append([float('nan'), t_data.shape[0], f_surface_tile_label, t_train_tile.shape[0], t_test_tile.shape[0], \
                      t_train_tile.shape[0] * 100.0 / t_data.shape[0], t_test_tile.shape[0] * 100.0 / t_data.shape[0], \
                      f_surface_train, f_surface_test, f_surface_train * 100.0 / f_surface_tile_label, f_surface_test * 100.0 / f_surface_tile_label])
    # write output stats to keep info on class repartition
    t_out = np.array(t_out)
    np.savetxt(os.path.join(s_out_tile, 'stats_par_classe.csv'), t_out, delimiter='\t', fmt=['%s']+['%d']*10, \
              header='Classe\tNbPoly\tSurface(ha)\tNbPolyTrain\tNbPolyTest\t%PolyTrain\t%PolyTest\tSurfaceTrain\tSurfaceTest\t%SurfaceTrain\t%SurfaceTest')
    # write shp files for train and test
    s_shp_train = os.path.join(s_out_tile, 'training.shp')
    #   for attribute AREA_HA we use the computed surface with GetArea() which is more reliable than the AREA_HA from input shp
    d_fields_train = {"CODE2" : t_train_tile[:, 1], "ID" : t_train_tile[:, 3], "AREA_HA" : t_train_tile[:, 2]}
    export_vector(s_shp_train, t_train_tile[:, 0], d_fields_train, driver="ESRI Shapefile", layer=s_layer_name, srs=o_srs)
    s_shp_test = os.path.join(s_out_tile, 'testing.shp')
    d_fields_test = {"CODE2" : t_test_tile[:, 1], "ID" : t_test_tile[:, 3], "AREA_HA" : t_test_tile[:, 2]}
    export_vector(s_shp_test, t_test_tile[:, 0], d_fields_test, driver="ESRI Shapefile", layer=s_layer_name, srs=o_srs)
    
    
    
    
def main():
    '''
    main
    '''
    # Parser creation
    parser = argparse.ArgumentParser(description='creation of labeled data for each S2 tile')
    # Args
    parser.add_argument('-img', '--img', metavar='[IMG_DIR]', help='Sentinel2 image directory (must contain a folder for each tile)', required=True)
    parser.add_argument('-label', '--label', metavar='[LABEL]',\
                        help='label vector file', required=True)
    
    parser.add_argument('-out', '--out', metavar='[OUT]', help='Output directory', required=True)
    # Command line parsing
    args = vars(parser.parse_args())
    s_img_dir = os.path.abspath(args['img'])
    s_label = os.path.abspath(args['label'])
    s_out = os.path.abspath(args['out'])
    os.system('mkdir -p {0}'.format(s_out))
    # get input images:
    t_img = get_image_list(s_img_dir)
     # reads footprints for each tile
    t_footprints = [read_tile_footprint(s_img ) for s_img in t_img]
    # find labels for each tile:
    t_labels = [label_data_by_tile(s_label, footprint) for footprint in t_footprints]
    # split labels into train and test
    [split_label(tile_polygons, o_srs, s_layer_name, s_out) for tile_polygons, o_srs, s_layer_name in t_labels]

if __name__ == '__main__':
    main()
