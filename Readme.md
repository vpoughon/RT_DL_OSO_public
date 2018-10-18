# Summary
- [Environnement](#env)
- [Create train test DB](#DB)
- [Rasterize labels](#rasterize)
- [Preprocessing](#preprocess)
- [File with information on references](#info)
- [Learning ](#learning)
- [Classification ](#classif)
- [Random Forest learning ](#RF_learning)
- [Random Forest classification ](#RF_classif)

# Environnement <a name="env"></a>
This code was developped and tested with following software environnement:
- CentOS 7
- python 3.5
- Tensorflow 1.9.0 (compiled with MKL-DNN support)
- Keras 2.2.0
- Horovod 0.13.10

The hardware environment is an HPC infrastructure: a cluster on a low latency network (Infiniband) with shared GPFS storage. Each node has 24 CPU cores and 120GB of memory. In total, there are 252 compute nodes available for a theoretical maximum of more than 6000 CPU cores. 

# Create train test DB <a name="DB"></a>

Label splitting between train and test datasets is performed with the script create_train_test_label.py. 

For instance:

python create_train_test_label.py -img /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract \
-label /work/OT/siaa/Work/RTDLOSO/partage/RPG2016_UA/learn_CP2014_RPG2016_UA.shp \
-out /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile

The output directory contains a folder per tile with train and test vector labels (shapefile format).

# Rasterize labels <a name="rasterize"></a>

python data_formatting_10m.py -img /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ \
-label /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile\
-out /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/RasterData_RPG2016_UA/

The output directory contains files "IdTuile_label_training.tif" and "IdTuile_label_testing.tif". It is label data rasterized in input image geometry.

# Preprocessing <a name="preprocess"></a>

The script Sentinel2_prepare_data.py is used to preprocess input data. It performs following steps:

- Computes and stores 3rd and 97th percentiles for ecah channel. It will be used for normalization
- Determines and stores pixels that contain labels for each class. It speeds up patch generation during learning.

Command line example:

python Sentinel2_prepare_data.py -img /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract 
 -label /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ RasterData_RPG2016_UA \
-out /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract

At the end of preprocessing, the directory of rasterized labels (RasterData_RPG2016_UA) will contain pickle files containing dictionnary of labels for each class and each tile. Labels are ready to be used in learning phase.


# File with information on references <a name="info"></a>

A file (info_references.txt) is created to store the name of each class, their arrangement in the output of the network and the colormap for the classification. This fil is stored in data directory (/work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/RasterData_RPG2016_UA). It is composed of 3 lines (excluding comments):
- Array representing class number arrangement: [11,12, 31, …, 221, 222]
- Dictionnary that binds class name and class number :  {«32» : «Foret feuilles persistantes», … }. This dictionnary is used to create the output Excel file representing evaluation results of classification
- Colormap as a dictionnary, that binds class ranking in the above array and its color in output land cover map: {« 0 » : [255,85,0], « 1 » :[255,255,127], … , « 16 » : [85,0,0]}

Thanks to this file, a change in class name or class number can be easily taken into account.

# Learning <a name="learning"></a>

The learning using MPI technology of CNES cluster is launched with the script: train_mpi.py

This script is a wrapper that executes the learning script Sentinel2_train_demo.py.
Inputs of train_mpi.py are:

    -rep:  directory containing input Sentinel2 image tiles (330 channels)
    -tile: tile list to be used in learning
    -raster: directory containing rasterized labeled data (and the file info_references.txt)
    -out: output directory that will contain saved model after each epoch and best model
    -recover: choice to recover from a previous learning (true/false), if a model is already present in output directory
    -nodes: cluster node number to use for learning. The number must be higher or equal to the number of input tiles. It is advised to use a multiple of the tile number.
    -epochs: number of epochs. 10000 patches are randomly generated per epoch (from all tiles and not per tile).


Command line example for a learning on 11 tiles:

python train_mpi.py -rep /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ \
-tile 'T30TWT T30TXQ T30TYN T30UXV T31TDJ T31TDN T31TEL T31TGK T31UDQ T31UDS T32ULU'\
-raster /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/RasterData_RPG2016_UA \
-out sentinel2_mlp_weights_Tall_11noeuds_instance1_30b_11t_batch44 \
-recover false -nodes 11 -epochs 50

Default hyperparameters are the followings:

- Learning rate = 0.0001 * node number
- Batch size = 4 * node number
- Loss: weighted categorical_crossentropy (weights defined according to class frequency in generated patches)
- Epoch number: 50


# Classification <a name="classif"></a>

Classification script is Sentinel2_Image_Classifier_qsub.py.

Inputs are:

    -rep: directory containing Sentinel2 tiles with 330 channels
    -tile: name of the tile to classify
    -model: directory of learning output, containing model created during learning
    -raster: directory containing rasterized labeled data (and the file info_references.txt)
    -vector: directory containing vector labeled data

This script applies to input tile the model learned during training. The tile is splitted in 512x512 pixels patches (can be changed according to available memory). They are extracted with a small overlap (16 pixels) because results may be bad near patch borders. Then a qsub job is launched to classify each patch. The final land cover map is obtained by merging of all classified patches.

Launching example:

python Sentinel2_ImageClassifier_qsub.py \
-rep /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract \
-tile T31TDN \
-model sentinel2_mlp_weights_Tall_11noeuds_instance1_30b_11t_batch44 \
-raster /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/RasterData_RPG2016_UA \
-vector /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile

As output, this script creates in output folder (arg -model):

- land cover map (with suffix _classif.tif). Pixel values represent class number
- color_map_tuile.tif: land cover map in RGB for better visualization
- Text files containing confusion matrix (confmat.csv), and contingency matrix (contmat.csv)
- Excel file (contmat_tuile.xls), containing confusion matrix in percent (1st sheet), and in hectares (2nd sheet)

Around 30 minutes are needed to classify a tile (depends on cluster availability)


# Random Forest learning <a name="RF_learning"></a>

The script RF_qsub.py performs training with Random Forest method, using Orfeo ToolBox application otbcli_TrainImagesClassifier. 

Inputs are:

    -rep: directory containing input Sentinel2 image tiles (330 channels)
    -tile: tile list to be used in learning
    -out: output directory that will contain learned model
    -vector: directory containing vector labeled data

Launch example:

python RF_qsub.py -rep /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract \
-tile 'T30TWT T30TXQ T30TYN T30UXV T31TDJ T31TDN T31TEL T31TGK T31UDQ T31UDS' \
-out /work/OT/siaa/Work/RTDLOSO/tests/RF/MultiTuiles \
-vector /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile

It performs following steps:
- Reserve qsub ressources needed for RF training
- Command line creation to execute otbcli_TrainImagesClassifier on inuput tiles, with following parameters:
    - -sample.vfn CODE2 (field containing class number in input vector data)
    - -classifier rf (to use Random Forest)
    - -classifier.rf.nbtrees 100 (100 trees max in the forest)
    - -classifier.rf.max 20 (tree depth)
    - -classifier.rf.cat 17 (17 classes)
    - -ram 10000 (allocated RAM in MB)
-Init environment and execute training

At the end of the training step, the model is created in a subfolder of the output folder.


# Random Forest classification <a name="RF_classif"></a>

Random Forest classification script is RF_classifier_qsub.py.

Inputs are:

    -rep: directory containing Sentinel2 tiles with 330 channels
    -tiles: name of the tile to classify
    -model: path to the model created during learning
    -out: output directory that will contain data generated by classification
    -raster: directory containing rasterized labeled data (and the file info_references.txt)
    -vector: directory containing vector labeled data


Launch example:

python RF_classifier_qsub.py \
-rep /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract \
 -model /work/OT/siaa/Work/RTDLOSO/tests/RF/MultiTuiles/T30TXQ /model_T30TXQ.rf \
-tiles T30TXQ \
 -out /work/OT/siaa/Work/RTDLOSO/tests/RF/MultiTuiles/T30TXQ \
 -raster /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/RasterData_RPG2016_UA \
 -vector /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile

It performs following steps:
- Reserve qsub ressources needed for RF classification
- Command line creation that will execute for each tile successively:
    - otbcli_ImageClassifier: performs classification
    - otbcli_ClassificationMapRegularization: regularizes land cover map, with structuring element of size 1 (arg -ip.radius 1)
    - script evaluate_classif.py that computes contingency and confusion matrices, created colormap and Excel file that contains statistical results
-Init environment and execute training

This script creates in output folder:
- land cover map (with suffix _classif.tif). Pixel values represent class number
- regularized map (with suffix _reg.tif)
- color_map_tuile.tif: land cover map in RGB for better visualization
- Text files containing confusion matrix (confmat.csv), and contingency matrix (contmat.csv)
- Excel file (contmat_tuile.xls), containing confusion matrix in percent (1st sheet), and in hectares (2nd sheet)


