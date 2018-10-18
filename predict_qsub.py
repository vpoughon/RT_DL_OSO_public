'''
Prediction sur une portion de tuile (typiquement 512x512)
exemple :
python predict_qsub.py /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ T31TDN sentinel2_mlp_weights_T31TDN_96cpu_30b_11t_lr0_0005_test3 ''
'''
# coding: utf-8

# In[1]:


import matplotlib
# Lancement en torque : il ne faut pas utiliser le backend par defaut (XWindow)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.colors import hsv_to_rgb
import cnes_data
import cnes_data.common as common
import sys
import os
import glob
import argparse



def process(s_img, PATCH_SIZE, s_raster_path, model_name, s_json, s_out_segmentation_map, s_out_confusion_map):
    
    label_map_vect, label_map, color_map = common.read_reference_info(os.path.join(s_raster_path, 'info_references.txt'))
    

    if not os.path.exists(model_name):
        raise("the model folder cannot be found")
    
    

    ## Load trained model
    from keras.models import Model
    from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, concatenate, Cropping2D
    from keras.optimizers import Adam, SGD
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler
    from keras import backend as K
    import json
    from keras.models import model_from_json
    
    import tensorflow as tf
    
    from data_formatting_10m import raster_to_3D_array, array_3D_to_raster
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    K.set_image_data_format('channels_first')
    
    
    def load_model(s_json, path_experiment, name_experiment, output_size):
        with open(s_json) as data_file:
            json_data = json.load(data_file)
            timesteps = 0
            b_lstm = 0
            # change input size
            for k in range(len(json_data["config"]["layers"])):
                if "batch_input_shape" not in  json_data["config"]["layers"][k]["config"]:
                    continue
                #print(json_data["config"]["layers"][k]["config"]["batch_input_shape"])
                if len(json_data["config"]["layers"][k]["config"]["batch_input_shape"]) == 5:
                    timesteps = json_data["config"]["layers"][k]["config"]["batch_input_shape"][1]
                    b_lstm = 1
                else:
                    timesteps += 1
                channels = json_data["config"]["layers"][k]["config"]["batch_input_shape"][-3]
                

        model = model_from_json(open(s_json).read())
        model.load_weights(path_experiment + '/' + name_experiment + '_' + 'best_weights.h5')
        return model, timesteps, channels, b_lstm
    
    unet_model, N_TIMESTEPS, N_CHANNELS, B_LSTM = load_model(s_json, model_name, model_name, PATCH_SIZE)
    
    
    # get patch
    #patch, gt = cnes_gen.get_validation_patches(y0, x0)
    patch, _, _ = raster_to_3D_array(s_img)
    # format and predict
    patch = np.array([patch])
    if B_LSTM:
        patch = np.array(np.split(patch, N_TIMESTEPS, axis=1)) # (11, 1, 30, 512, 512)
        patch = np.transpose(patch, (1, 0, 2, 3, 4)) # (1, 11, 30, 512, 512)
    else:
        patch = np.split(patch, N_TIMESTEPS, axis=1) # [(1, 30, 512, 512)] x11
    pred_patches = unet_model.predict(patch)
    # format for evaluation and display
    pred_patches = np.reshape(pred_patches,(pred_patches.shape[0], PATCH_SIZE,PATCH_SIZE,pred_patches.shape[-1]))
    pred_patch_argmax = np.argmax(pred_patches,axis=3)# todo: change :-1
    preds_patches_idx = np.zeros(pred_patch_argmax.shape)
    preds_patch_max = np.max(pred_patches,axis=3) # probas max, entre 0 et 1
    for k in range(len(label_map_vect)):
        k_argmax = (pred_patch_argmax==k).astype('float32')
        preds_patches_idx += k_argmax*label_map_vect[k]
    # write outputs: preds_patches_idx and preds_patch_max
    array_3D_to_raster(np.uint8(preds_patches_idx), s_out_segmentation_map)
    array_3D_to_raster(np.float32(preds_patch_max), s_out_confusion_map)
    return 0


def main():
    '''
    main
    '''
    # Parser creation
    parser = argparse.ArgumentParser(description='Sentinel2 classifier')
    # Args
    #inputs : x0, y0, pad, PATCH_SIZE, s_out_segmentation_map, s_out_confusion_map, model_name
    parser.add_argument('-img', '--img', metavar='[IMG]', help='Path to the raster subtile to classify', required=True)
    parser.add_argument('-patch', '--patch', metavar='[PATCH_SIZE]', help='Patch size (typically 512)', required=True)
    parser.add_argument('-model', '--model', metavar='[MODEL]', help='Model to use for classification', required=True)
    parser.add_argument('-raster', '--raster', metavar='[RASTER_DATA]', help='Directory containing rasterized labeled data', required=True)
    parser.add_argument('-json', '--json', metavar='[JSON_ARCHI]', help='Fichier json representant  architecture reseau', required=True)
    parser.add_argument('-out_seg', '--out_seg', metavar='[OUTPUT_SEGMENTATION_MAP]', help='Name for the output segmentation map', required=True)
    parser.add_argument('-out_conf', '--out_conf', metavar='[OUTPUT_CONFUSION_MAP]', help='Name for the output confusion map (containing proba for the class)', required=True)
    # Command line parsing
    args = vars(parser.parse_args())
    s_img = args['img']
    u_patch = int(args['patch'])
    s_raster_path = args['raster']
    s_model = args['model']
    s_json = args['json']
    s_out_seg = os.path.abspath(args['out_seg'])
    s_out_conf = os.path.abspath(args['out_conf'])
    process(s_img, u_patch, s_raster_path, s_model, s_json, s_out_seg, s_out_conf)
    

if __name__ == '__main__':
    main()


