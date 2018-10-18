# coding: utf-8
'''

exemple :
python Sentinel2_ImageClassifier_qsub.py -rep /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ -tile T31TDN \
       -model sentinel2_mlp_weights_T31TDN_4noeuds_instance1_30b_11t_batch16_adamlr0_0001_noweight_test23 \
       -raster /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/RasterData_RPG2016_UA \
       -vector /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile

'''


# In[1]:

import sys
import os
import glob
import subprocess
import shutil
import matplotlib
# Lancement en torque : il ne faut pas utiliser le backend par defaut (XWindow)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import argparse
from matplotlib.colors import hsv_to_rgb
import cnes_data
import cnes_data.common as common
from tqdm import tqdm
from data_formatting_10m import raster_to_3D_array, array_3D_to_raster

import job_array

PATCH_SIZE = 512
in_notebook = False
try:
    get_ipython
    in_notebook = True
except:
    print("Running in terminal...")


# Parser creation
parser = argparse.ArgumentParser(description='Sentinel2 classifier')
# Args
parser.add_argument('-rep', '--rep', metavar='[REP_IMAGE]', help='', required=True)
parser.add_argument('-tile', '--tile', metavar='[ID_TILE]', help='Tile ID', required=True)
parser.add_argument('-model', '--model', metavar='[MODEL_DIR]', help='Directory containing the learned model', required=True)
parser.add_argument('-raster', '--raster', metavar='[RASTER_DATA]', help='Directory containing rasterized labeled data', required=True)
parser.add_argument('-vector', '--vector', metavar='[VECTOR_DATA]', help='Directory containing vector reference data by tile', required=True)
# Command line parsing
args = vars(parser.parse_args())
DB_PATH = os.path.abspath(args['rep'])
TILE_NAME = args['tile']
model_name = args['model']
s_raster_path = args['raster']
s_vector_path = args['vector']
OTB_BIN_PATH = ''


if not os.path.exists(model_name):
    raise("the model folder cannot be found")


# In[3]:
# reads LASS_ID_SET, label_map, color_map in info_references.txt
label_map_vect, label_map, color_map = common.read_reference_info(os.path.join(s_raster_path, 'info_references.txt'))

s_out_dir = os.path.join(model_name, TILE_NAME)
os.system('mkdir -p {0}'.format(s_out_dir))
# Get testing tile, test annotations and min/max channels
colormap_file = os.path.join(s_raster_path, 'color_map.txt')
test_gt_shp_file =  os.path.join(s_vector_path, TILE_NAME, 'testing.shp')
min_channels_filename = glob.glob(os.path.join(DB_PATH, TILE_NAME + '/*min_channels.npy'))
min_channels = np.load(min_channels_filename[0])
max_channels_filename = glob.glob(os.path.join(DB_PATH, TILE_NAME + '/*max_channels.npy'))
max_channels = np.load(max_channels_filename[0])

out_segmentation_map = os.path.join(s_out_dir, model_name + "_classif.tif")
out_confusion_map = os.path.join(s_out_dir, model_name + "_confusion.tif")


# In[4]:


## Load trained model
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, concatenate, Cropping2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import json
from keras.models import model_from_json

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

K.set_image_data_format('channels_first')


def load_model(path_experiment, name_experiment, output_size):
    with open(path_experiment + '/' + 'architecture.json') as data_file:
        json_data = json.load(data_file)
        timesteps = 0
        b_lstm = 0
        # change input size
        for k in range(len(json_data["config"]["layers"])):
            if "batch_input_shape" not in  json_data["config"]["layers"][k]["config"]:
                continue
            print(json_data["config"]["layers"][k]["config"]["batch_input_shape"])
            if len(json_data["config"]["layers"][k]["config"]["batch_input_shape"]) == 5:
                timesteps = json_data["config"]["layers"][k]["config"]["batch_input_shape"][1]
                b_lstm = 1
            else:
                timesteps += 1
            json_data["config"]["layers"][k]["config"]["batch_input_shape"][-2] = output_size
            json_data["config"]["layers"][k]["config"]["batch_input_shape"][-1] = output_size
            channels = json_data["config"]["layers"][k]["config"]["batch_input_shape"][-3]
            
        # change concatenation size (only if unet+mlp, otherwise will be ignored)
        for k in range(len(json_data["config"]["layers"])):
            if "target_shape" not in  json_data["config"]["layers"][k]["config"]:
                continue
            if len(json_data["config"]["layers"][k]["config"]["target_shape"]) < 3:
                continue
            print(json_data["config"]["layers"][k]["config"]["target_shape"])
            if json_data["config"]["layers"][k]["config"]["target_shape"][1] == json_data["config"]["layers"][k]["config"]["target_shape"][2]:
                json_data["config"]["layers"][k]["config"]["target_shape"][1] = output_size
                json_data["config"]["layers"][k]["config"]["target_shape"][2] = output_size

        #json_data["config"]["layers"][0]["config"]["batch_input_shape"][2] = output_size
        #json_data["config"]["layers"][0]["config"]["batch_input_shape"][3] = output_size
        last_lay = len(json_data["config"]["layers"]) - 3
        print(json_data["config"]["layers"][last_lay]["config"]["target_shape"][1])
        json_data["config"]["layers"][last_lay]["config"]["target_shape"][1] = output_size * output_size

    with open(path_experiment + '/temp_architecture.json', 'w') as outfile:
        json.dump(json_data, outfile)

    model = model_from_json(open(path_experiment + '/temp_architecture.json').read())
    #model.load_weights(path_experiment + '/' + name_experiment + '_' + 'best_weights.h5')
    return model, timesteps, channels, b_lstm

unet_model, N_TIMESTEPS, N_CHANNELS, B_LSTM = load_model(model_name, model_name, PATCH_SIZE)

s_rep_qsub = os.path.abspath(os.path.join(s_out_dir, 'qsub'))
os.system('mkdir -p {0}'.format(s_rep_qsub))

# load tile and annotations for testing
cnes_gen = cnes_data.CnesGeneratorSentinel(TILE_NAME, DB_PATH, s_raster_path)
cnes_gen.set_multi_class(label_map_vect)
cnes_gen.set_patch_size(PATCH_SIZE)


# cut tile in patches for prediction

# patch VP
#u_nb_col, u_nb_row = cnes_gen.get_tile_size()
u_nb_row, u_nb_col = cnes_gen.get_tile_size()
# fin patch VP
pad = 20
hpad = pad // 2
NB_SUB_Y = ((u_nb_row+pad) // (PATCH_SIZE-pad)) + 1
NB_SUB_X = ((u_nb_col+pad) // (PATCH_SIZE-pad)) + 1

# output tile
tile_out = np.zeros((u_nb_row+PATCH_SIZE, u_nb_col+PATCH_SIZE))
confusion_tile = np.zeros((u_nb_row+PATCH_SIZE, u_nb_col+PATCH_SIZE))

centers_x = (np.arange(NB_SUB_X)*(PATCH_SIZE - pad) + PATCH_SIZE/2-pad).astype('int32')
centers_y = (np.arange(NB_SUB_Y)*(PATCH_SIZE - pad) + PATCH_SIZE/2-pad).astype('int32')
print(centers_x)
print(centers_y)

#       initialisation des fichiers du jobArray:
o_JA = job_array.jobArray("Classif", s_rep_qsub, CLUSTER_CHECK_TIME=30)
jobArrayScript = o_JA.getScriptFile()
jobArrayParamFile = o_JA.getParamFile()
rep_qsub_out = o_JA.getQsubRepOut()
suffixe = o_JA.getSuffixe()
jobArrayParamList = []
u_count_total = 0
s_json = model_name + '/temp_architecture.json'
# generation des classifs et confusion maps pour chaque sous tuile, via jobs qsub
for x in centers_x:
    for y in centers_y:
        print('{0}, {1}'.format(x,y))
        # get patch
        patch = cnes_gen.get_validation_patches_without_gt(y-(PATCH_SIZE)//2,x-(PATCH_SIZE)//2)
        # write patch that will be processed in a job qsub
        #print(patch[0, ...].shape)
        s_img = os.path.join(s_rep_qsub, "patch_{}_{}.tif".format(x, y))
        s_out_seg = os.path.join(s_rep_qsub, "outseg_{}_{}.tif".format(x, y))
        s_out_conf = os.path.join(s_rep_qsub, "outconf_{}_{}.tif".format(x, y))
        if not os.path.isfile(s_img):
            array_3D_to_raster(patch[0, ...], s_img)
        if not os.path.isfile(s_out_seg) or not os.path.isfile(s_out_conf):
            s_cmd = "python predict_qsub.py -img "+ s_img +" -patch " + str(PATCH_SIZE) + " -raster " + s_raster_path + " -model " + model_name + " -json " + s_json +\
                    " -out_seg " + s_out_seg + " -out_conf " +  s_out_conf + "\n"
            jobArrayParamList.append(s_cmd)
            u_count_total += 1


cnes_gen = None # pour liberer la memoire, car subprocess appele pour JobArray veut autant de memoire libre que de memoire occupee par le script qui l'appelle (OSError: [Errno 12] Cannot allocate memory)

if u_count_total > 0:
    # on ecrit les parametres de lancement de chaque sous tuile :
    jobArrayparamfile = open(jobArrayParamFile, "w")
    jobArrayparamfile.writelines(jobArrayParamList)
    jobArrayparamfile.close()
    #creation et execution d'un fichier JobArrayScript  :
    filescript=open(jobArrayScript, 'w')
    ncpu = 12
    s_mem = "32000MB"
    if B_LSTM:
        ncpu = 24 # otherwise number cpu used exceed 12 in LSTM networks
        s_mem = "64000MB"
    filescript.write("#!/bin/bash \n#PBS -J 1-"+str(u_count_total)+":1\n#PBS -l walltime=00:10:00 \n#PBS -l select=1:ncpus=" + str(ncpu) + ":mem=" + s_mem +":os=rh7 \n \
    #PBS -N Classif \n#PBS -o "+rep_qsub_out+" \n#PBS -e "+rep_qsub_out+" \n")
    # script d'init :
    filescript.write(". /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_MKL.sh\n")
    filescript.write("cd {}\n".format(os.getcwd()))
    # fichier contenant les parametres python :
    filescript.write("input="+jobArrayParamFile+" \n")
    # lecture de chaque ligne du fichier de parametres :
    filescript.write("params=$(sed -n ${PBS_ARRAY_INDEX}p $input) \n")
    # execution du script python pour chaque produit :
    filescript.write("$params \n")
    filescript.close()
    returnCode = os.system("chmod +x "+jobArrayScript)
    returnCode = o_JA.execute()
    if returnCode != 0:
        raise Exception('Error during execution of sub tile classification')

    
# lecture des classif et confusion maps
for x in centers_x:
    for y in centers_y:
        s_out_seg = os.path.join(s_rep_qsub, "outseg_{}_{}.tif".format(x, y))
        s_out_conf = os.path.join(s_rep_qsub, "outconf_{}_{}.tif".format(x, y))
        preds_patches_idx, _, _ = raster_to_3D_array(s_out_seg)
        preds_patch_max, _, _ = raster_to_3D_array(s_out_conf)
        # add patch to tile
        tile_out[(y-PATCH_SIZE//2+pad):(y+PATCH_SIZE//2), 
             (x-PATCH_SIZE//2+pad):(x+PATCH_SIZE//2)] = preds_patches_idx[hpad:-hpad,hpad:-hpad]
        confusion_tile[(y-PATCH_SIZE//2+pad):(y+PATCH_SIZE//2), 
             (x-PATCH_SIZE//2+pad):(x+PATCH_SIZE//2)] = preds_patch_max[hpad:-hpad,hpad:-hpad]

tile_out = tile_out[hpad:u_nb_row+hpad, hpad:u_nb_col+hpad]
confusion_tile = confusion_tile[hpad:u_nb_row+hpad, hpad:u_nb_col+hpad]
# save tile out


# clean up temporary files
for fic in glob.glob(os.path.join(s_rep_qsub, "outseg_*.tif")):
    os.remove(fic)
for fic in glob.glob(os.path.join(s_rep_qsub, "outconf_*.tif")):
    os.remove(fic)
for fic in glob.glob(os.path.join(s_rep_qsub, "patch_*.tif")):
    os.remove(fic)
for fic in glob.glob(os.path.join(s_rep_qsub, "temp_architecture_*.json")):
    os.remove(fic)


# Debug tool : Plot specific area of the tile to make sure that everything looks fine
if in_notebook:
    y_start = 2000
    y_end = 2300
    x_start = 2000
    x_end = 2300
    channels = [100,101,102]

    import cv2
    pred_crop = tile_out[y_start:y_end,x_start:x_end]
    lut = np.zeros((1,256,3))
    ct = 0
    for cl in label_map_vect:
        lut[0,cl,0] = color_map[ct][0]
        lut[0,cl,1] = color_map[ct][1]
        lut[0,cl,2] = color_map[ct][2]
        ct += 1

    # get cropped images
    gt_crop, img_crop = cnes_gen.get_display_patch(y_start,y_end,x_start,x_end,channels)
    pred_crop = tile_out[y_start:y_end,x_start:x_end]

    # color gt and prediction
    pred_crop = cv2.cvtColor(pred_crop.astype("uint8"), cv2.COLOR_GRAY2RGB)
    pred_crop2 = cv2.LUT(pred_crop, lut).astype("uint8")
    gt_crop = cv2.cvtColor(gt_crop.astype("uint8"), cv2.COLOR_GRAY2RGB)
    gt_crop2 = cv2.LUT(gt_crop, lut).astype("uint8")
    img_crop = (img_crop/10).astype('uint8')

    fig = plt.figure(figsize=(24, 24))
    plt.subplot(131)
    plt.imshow(img_crop)
    plt.subplot(132)
    plt.imshow(pred_crop2)
    plt.subplot(133)
    plt.imshow(gt_crop2)
    plt.show() 


## evaluate and save data

import subprocess
import sys
import gdal
import gdalconst

cnes_gen = 0
tile_out_color = None
tile_gt_color = None
tile_bands = None
# get metadata and save result
gtif_sample = os.path.join(DB_PATH , TILE_NAME + '/Sentinel2_ST_GAPFIL.tif')
#t_img, o_proj, o_geo = raster_to_3D_array( gtif_sample)
o_image = gdal.Open(gtif_sample)
if o_image is None:
    print("couldn't open input dataset {0}".format(gtif_sample))
o_proj = o_image.GetProjection()
o_geo = o_image.GetGeoTransform()
o_image = None

print(tile_out.shape)
print(out_segmentation_map)
array_3D_to_raster(tile_out, out_segmentation_map, o_proj, o_geo)
array_3D_to_raster(confusion_tile, out_confusion_map, o_proj, o_geo)

# write associated metadatas
out_metadata = out_segmentation_map + ".aux.xml"
print(out_metadata)
file = open(out_metadata ,'w')  
file.write('<PAMDataset><SRS>') 
file.write(o_proj) 
file.write('</SRS></PAMDataset>')
file.close() 


# generate confusion matrix
from data_formatting_10m import raster_to_3D_array, array_3D_to_raster
import subprocess
import sys
import gdal
import gdalconst

# enchainement avec evaluate_classif:
t_cmd = [". /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_MKL.sh && python /work/OT/siaa/Work/RTDLOSO/scripts/evaluate_classif.py", 
                    "-img", out_segmentation_map,
                    "-label", test_gt_shp_file,
                    "-out", s_out_dir,
                    "-cmap", os.path.join(s_raster_path, 'color_map.txt')]
s_cmd = ' '.join(t_cmd)
print(s_cmd)
process = subprocess.Popen([s_cmd], shell=True,
                          stdout=subprocess.PIPE)
# write logfile
ct = 0
logfile = open(os.path.join(s_out_dir, 'logfile_evaluation_classif.txt'), 'w')
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line.decode('utf-8'))
    ct += 1
    if ct > 1000:
        break
    logfile.write(line.decode('utf-8')) # remove decode for python 2.7
process.wait()
logfile.close()



