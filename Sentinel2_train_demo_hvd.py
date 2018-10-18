
# coding: utf-8

# In[1]:

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


import horovod.keras as hvd
import horovod.tensorflow as hvd_tf

hvd.init()
assert hvd_tf.mpi_threads_supported()

# Make sure MPI is not re-initialized.
import mpi4py.rc
mpi4py.rc.initialize = False

from mpi4py import MPI
assert hvd.size() == MPI.COMM_WORLD.Get_size()

from model_definitions import *
import tensorflow as tf

import keras
from keras import backend as K
from keras.utils import plot_model
import importlib

from patch_display import make_mosaic_result, save_img, format_patches_for_display, format_patches_for_display_colormap
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import cnes_data
import cnes_data.common as common

import ml_metrics

#os.environ['NCCL_P2P_DISABLE'] = '1'
#os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

use_background_layer = False # use the background layer in the model
PATCH_SIZE = 64
N_TIMESTEPS = 11 #33
N_CHANNELS = 30 #10 
NB_PATCH_PER_EPOCH = 10000
nb_valid_patch = 500
batch_size_weight_estimation = 32
nb_iterations_weight_estimation = 200 # could be done accurately enough on 100 iterations.
b_lstm = 0 # 1 to use LSTM model or 0 to use network duplication over timesteps

use_contours = False
if use_contours:
    use_background_layer = True
    PATCH_SIZE = 64
use_rf_annotations = False


in_notebook = False
try:
    get_ipython
    in_notebook = True
except:
    print("Running in terminal...")



# Parser creation
parser = argparse.ArgumentParser(description='Sentinel2 hvd training')
# Args
parser.add_argument('-rep', '--rep', metavar='[REP_IMAGE]', help='', required=True)
parser.add_argument('-tile', '--tile', metavar='[ID_TILE]', help='Tile ID', required=True)
parser.add_argument('-out', '--out', metavar='[OUT_DIR]', help='Output directory that will contain the learned model', required=True)
parser.add_argument('-recover', '--recover', metavar='[RECOVER]', help='true/false to  allow to start training from a saved model', required=True)
parser.add_argument('-raster', '--raster', metavar='[RASTER_DATA]', help='Directory containing rasterized labeled data', required=True)
parser.add_argument('-epochs', '--epochs', metavar='[EPOCH_NUMBER]', help='Number of epochs', default=75, required=False)
# Command line parsing
args = vars(parser.parse_args())
DB_PATH = os.path.abspath(args['rep'])
t_tile_name = args['tile'].split(' ')
resume_training = args['recover'] == "true"
name_experiment = os.path.normpath(args['out'])
s_raster_dir = args['raster']
NUM_EPOCHS = int(args['epochs'])
use_hyperas_optim = 0
if not os.path.exists(DB_PATH):
    print("Dataset file {} does not exist!".format(DB_PATH))

        
if not os.path.exists(name_experiment):
    try:
        os.makedirs(name_experiment)
    except FileExistsError:
        pass

snapshot_file_name = name_experiment+'/'+name_experiment+'_best_weights.h5'
if resume_training:
    if not os.path.exists(snapshot_file_name):
        raise Exception("ERROR: Trying to resume from non-existing snapshot {}".format(snapshot_file_name))
        
    print("Training will resume from snapshot {}".format(snapshot_file_name))



def data_generator(batch_size, gen, epoch_len, temporal_seq = 0, use_background_layer=True, u_nb_tile=1, b_lstm=0):
    """ Generates training sequences on demand
    """
    cnes_gen_util = cnes_data.CnesGen10mUtilHvd(gen, PATCH_SIZE)
    
    while True:
        for idx in range(epoch_len):
            # no need to exchange patchs between workers if just one tile as input
            if u_nb_tile == 1:
                X, Y = gen.generate_train_patch_fast(batch_size)
            else:
                X, Y = cnes_gen_util.generate_train_patch_using_sharing(batch_size)
            # X : shape (4, 330, 64, 64) (if batch_size=4)

            Y = np.reshape(Y,(Y.shape[0], Y.shape[1],Y.shape[2]*Y.shape[3]))
            Y = np.transpose(Y,(0,2,1))
            if temporal_seq > 0 and not b_lstm:
                X = np.split(X,temporal_seq,axis=1)
            if temporal_seq > 0 and b_lstm:
                X = np.transpose(np.array(np.split(X, temporal_seq, axis=1)), (1,0,2,3,4)) # batch_size, nb_dates, nb_channels, 64,64) # (4, 11, 30, 64, 64)
            #print(X.shape)
            yield (X, np.array(Y))

def data_generator_without_mpi4py(batch_size, gen, epoch_len, temporal_seq = 0, use_background_layer=True, b_lstm=0):
    """ Generates training sequences on demand
    """
    while True:
        for idx in range(epoch_len):
            X, Y = gen.generate_train_patch_fast(batch_size)
            Y = np.reshape(Y,(Y.shape[0], Y.shape[1],Y.shape[2]*Y.shape[3]))
            Y = np.transpose(Y,(0,2,1))
            if temporal_seq > 0 and not b_lstm:
                X = np.split(X,temporal_seq,axis=1)
            if temporal_seq > 0 and b_lstm:
                X = np.transpose(np.array(np.split(X, temporal_seq, axis=1)), (1,0,2,3,4)) # batch_size, nb_dates, nb_channels, 64,64) # (4, 11, 30, 64, 64)
            yield (X, np.array(Y))

def compute_class_stats_train(gen, batch_size, nb_iterations):
        # patch VP : il ne faut pas se baser sur la frequence dans les annotations ou dans les images, mais de celle dans les patchs generes
        # la generation des patchs fait en sorte d'equilibrer les classes, et la frequence des classes dans les patchs n'est pas la meme que dans les annotations
        cnes_gen_util = cnes_data.CnesGen10mUtilHvd(gen, PATCH_SIZE)
        class_stats = np.zeros(len(CLASS_ID_SET))
        for i in range(nb_iterations):
            if hvd.rank() == 0 and i % 20 == 0:
                print("{} / {}".format(i, nb_iterations))
            patch, gt_patch = cnes_gen_util.generate_train_patch_using_sharing(batch_size)
            for ct in range(len(CLASS_ID_SET)):
                positive_positions = np.where(gt_patch[:,ct,...] > 0)
                class_stats[ct] += len(positive_positions[0])
        return class_stats

# Draw a patch of samples to visually evaluate the network state
class DrawEstimates(keras.callbacks.Callback):
    def set_data(self, data, mask,name_experiment='.'):
        self.samples = data
        self.masks = mask
        self.preds = np.zeros(mask.shape)
        self.epoch = 0
        self.name_experiment = name_experiment

    def on_epoch_begin(self, batch, logs={}):
        self.epoch += 1
        print("") 

    def on_epoch_end(self, batch, logs={}):
        self.preds = self.model.predict(self.samples, batch_size=32, verbose=2)
        # draw !
        self.draw_and_save()
    
    def draw_and_save(self):
        # save
        samples = np.concatenate(self.samples, axis=1)
        masks = self.masks[:self.preds.shape[0],:self.preds.shape[1],:self.preds.shape[2]]
        plot_patch, gt_patches_viz, preds_patches_viz = format_patches_for_display_colormap(samples, masks, self.preds, 
                                                                                   input_ch=[2,1,0], input_gain=1, colormap=color_map) 
        save_img(make_mosaic_result(plot_patch, gt_patches_viz, preds_patches_viz),
                  name_experiment + '/sample_results_' + str(self.hvd_rank) + "_" + str(self.epoch))
    def draw(self):
        samples = np.concatenate(self.samples, axis=1)
        plot_patch, gt_patches_viz, preds_patches_viz = format_patches_for_display_colormap(samples, self.masks, self.preds, 
                                                                                   input_ch=[2,1,0], input_gain=5, colormap=color_map) 
        
    def __init__(self, hvd_rank):
        self.hvd_rank = hvd_rank




class PrintClassStats(keras.callbacks.Callback):
    epoch = 0
    def set_gen(self, gen):
        self.cnes_gen = gen
        
    def on_epoch_begin(self, batch, logs={}):
        self.epoch += 1

    def on_epoch_end(self, batch, logs={}):
        # get estimates
        stats = self.cnes_gen.get_running_stats()
        
        print("stats at rank {} : {}".format(hvd.rank(), stats))
        
        stats_mat = np.zeros((len(CLASS_ID_SET)+1, 2), np.float32)
        stats_mat[0,1] = stats[0]
        idx = 1
        for cid in CLASS_ID_SET:
            stats_mat[idx,0] = cid
            if cid in stats:
                stats_mat[idx,1] = stats[cid]
            idx+=1
        
        print("Gathering stats from all MPI instances, rank {}".format(hvd.rank()))
        all_stats = hvd.allgather(stats_mat) #comm.gather(stats, root=0)
        total_px = 0
        
        if hvd.rank() == 0:
            print("Epoch {} class freqs:".format(self.epoch))
            class_stats = {class_id:0 for class_id in CLASS_ID_SET}
            for class_id in CLASS_ID_SET:
                #print("Data for class {}: {}".format(class_id, all_stats[all_stats[:,0] == class_id, :]))
                px_class = np.sum(all_stats[all_stats[:,0] == class_id, 1])
                class_stats[class_id] += px_class
                total_px += px_class
        
            non_annot_px = np.sum(all_stats[all_stats[:,0] == 0, 1])
            total_px += non_annot_px
            print("Non annotated pixels : {}".format(non_annot_px))
            for class_id in class_stats:
                print("Class {} count = {}, freq {:.5f}%".format(class_id, class_stats[class_id], class_stats[class_id]/total_px*100))

class clsvalidation_kappa(keras.callbacks.Callback):  #inherits from Callback
    def __init__(self, name_experiment, validation_data=()):
        super(keras.callbacks.Callback, self).__init__()
        self.X_val, self.y_val = validation_data  #tuple of validation X and y
        self.best = 0.0
        self.sess = K.get_session()
        # Specifying the basedir of Tensorboard
        self.summaries_dir = os.path.join(name_experiment, 'logs_tensorboard')
        # Creating the tensor that will hold the loss value
        # computed by the metrics through keras API
        self.t_fscore_sum = []
        for u_class in range(len(CLASS_ID_SET)):
            #fscore_tensor = K.tf.placeholder(K.tf.float32, name='fscore_tensor_{}'.format(CLASS_ID_SET[u_class]))
            # Adding a summary tracking the value of the precedent
            # tensor under the name 'fscore'. This name will
            # be the name of the section in the Tensorboard dashboard
            self.t_fscore_sum.append(K.tf.summary.scalar('fscore_{}'.format(CLASS_ID_SET[u_class]), K.tf.placeholder(K.tf.float32, name='fscore_tensor_{}'.format(CLASS_ID_SET[u_class]))))
        self.sum_writer = K.tf.summary.FileWriter(self.summaries_dir + '/fscore')
        
    def on_epoch_end(self, epoch, logs={}):
        pred_patches = self.model.predict(self.X_val, verbose=0)
        pred_patch_argmax = np.argmax(pred_patches, axis=-1)
        gt_patches_argmax = np.argmax(self.y_val, axis=-1)
        gt_patche_max = np.max(self.y_val, axis=-1) # 1 si label, 0 si pas de label
        # on supprime les pixels qui n'ont pas d'annotation
        gt_patches_argmax = gt_patches_argmax[gt_patche_max == 1]
        pred_patch_argmax = pred_patch_argmax[gt_patche_max == 1]
        
        y_pred = np.ravel(pred_patch_argmax) # toutes les predictions de chaque pixel (4096 - no labeled pixels) des n patchs dans un vecteur (valeurs entre 0 et 15)
        y_true = np.ravel(gt_patches_argmax)
        # gathering from all MPI instances
        y_pred = hvd.allgather(y_pred)
        y_true = hvd.allgather(y_true)
        kappa = ml_metrics.quadratic_weighted_kappa(y_true, y_pred)
        val_accuracy = float(len(np.where(y_pred == y_true)[0])) / float(len(y_pred))

        if hvd.rank() == 0:
            if kappa > self.best:
                self.best = kappa
            print('Epoch %d Val Accuracy: %f | Val Kappa: %f | Best Val Kappa: %f \n' % (epoch, val_accuracy, kappa, self.best))
            # precision, recall for each class:
            for u_class in range(len(CLASS_ID_SET)):
                y_pred_class = y_pred[y_pred == u_class]
                y_true_class = y_true[y_pred == u_class]
                f_tp = float(len(np.where(y_pred_class == y_true_class)[0]))
                f_tp_plus_fp = float(len(np.where(y_pred == u_class)[0]))
                f_tp_plus_fn = float(len(np.where(y_true == u_class)[0]))
                if f_tp_plus_fp != 0:
                    f_prec = f_tp / f_tp_plus_fp
                else:
                    f_prec = 0.0
                if f_tp_plus_fn != 0:
                    f_recall = f_tp / f_tp_plus_fn
                else:
                    f_recall = 0.0
                f_Fscore = 0.0
                if f_recall != 0.0 and f_prec != 0.0:
                    f_Fscore = 2 * f_recall * f_prec / (f_recall + f_prec)
                print('Class {}  , precision - recall - Fscore: {}\t{}\t{}'.format(CLASS_ID_SET[u_class], f_prec, f_recall, f_Fscore))
                # to visualize in TensorBoard
                u_class_id = CLASS_ID_SET[u_class]
                self.sum_writer.add_summary(self.sess.run(self.t_fscore_sum[u_class], \
                                            feed_dict={'fscore_tensor_{}:0'.format(u_class_id): f_Fscore}), epoch)


if in_notebook:
    importlib.reload(cnes_data)


# In[8]:
# reads LASS_ID_SET, label_map, color_map in info_references.txt
CLASS_ID_SET, label_map, color_map = common.read_reference_info(os.path.join(s_raster_dir, 'info_references.txt'))

# load tile and annotations
# possibility to have several tile in input. They will be given to the various nodes:
u_nb_tile = len(t_tile_name)
if u_nb_tile > hvd.size():
    raise(Exception('Error, to many tile in input. The number of tiles cannot be higher than the number of nodes'))
u_index = int(hvd.rank() * u_nb_tile //  hvd.size())

print('worker {} Tile {}'.format(hvd.rank(), t_tile_name[u_index]))
cnes_gen = cnes_data.CnesGeneratorSentinel(t_tile_name[u_index], DB_PATH, s_raster_dir, 1, 0) # new reference data 2016

cnes_gen.compute_valid_positions_multi_tiles(t_tile_name, hvd.size())

cnes_gen.set_multi_class(CLASS_ID_SET)
cnes_gen.set_patch_size(PATCH_SIZE)
cnes_gen.enable_contours_channel(use_contours)
cnes_gen.enable_running_stats(True)

# save min and max for inference
min_channels, max_channels = cnes_gen.get_min_max_channels()


if hvd.rank() == 0:
    np.save(name_experiment + '/min_channels', min_channels)
    np.save(name_experiment + '/max_channels', max_channels) 

if use_rf_annotations:
    cnes_gen.set_dense_annotations(rf_path)


#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config.gpu_options.visible_device_list = str(hvd.local_rank())
#sess = tf.Session(config=config)


config = tf.ConfigProto(inter_op_parallelism_threads=4) # necessary for Unet only model to limit it to 4 (error if omitted)
session = tf.Session(config=config)
K.set_session(session)

K.set_image_data_format('channels_first')
print("Data format is: " + K.image_data_format())

patch, gt = cnes_gen.generate_train_patch_fast(1)
num_input_channels = patch.shape[1]
num_output_channels = gt.shape[1]

if hvd.rank() == 0:
    print("Class weight estimation on {} iterations with batch size={} on each worker".format(nb_iterations_weight_estimation, batch_size_weight_estimation))
class_weight = compute_class_stats_train(cnes_gen, batch_size_weight_estimation, nb_iterations_weight_estimation) # with patch sharing (longer but more accurate)
all_class_weight = hvd.allgather(class_weight)
all_class_weight = np.sum(np.split(all_class_weight, hvd.size()), axis=0)
# weight = 1 / percent_label_class_in_patchs
all_class_weight = np.divide(all_class_weight * 100.0, np.sum(all_class_weight))
class_weight = np.divide(1, all_class_weight, out=np.zeros_like(all_class_weight), where=all_class_weight!=0)

# weight normalization to make their sum equal to the numer of classes (16):
class_weight = class_weight * len(CLASS_ID_SET) / np.sum(class_weight)



def train_model(alpha, beta, base_lr, lr_reduce_auto, prefix_sub_experiment="hyperas"):
    print("Using alpha={}, beta={}, base_lr={}, lr_reduce_auto={}".format(alpha, beta, base_lr, lr_reduce_auto))

    print("Class weights {}".format(class_weight))
    BATCH_SIZE = 4  # for total batch of size 4 * hvd.size()
    # Horovod: adjust learning rate based on number of GPUs.
    opt = keras.optimizers.Adam(lr=base_lr * hvd.size())
    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)
    
    t_mlp_shallow = [200,100,50] # 3 layers
    t_mlp_deep = [200] *3 + [100] * 3 + [50] * 3 # 9 layers
    if not b_lstm:
        # Unet + MLP (FG-UNET) with model duplication over timesteps (Total params: 525,997)
        unet_model = get_unet_mlp_ts(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE,[200,100,50], opt, class_weight)
        # only MLP
        #unet_model = get_sequential_mlp(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE, t_mlp_shallow, opt, class_weight)
        # Deep MLP
        #unet_model = get_sequential_mlp(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE, t_mlp_deep, opt, class_weight)
        # only Unet
        #unet_model = get_unet_only(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE, opt, class_weight)
    else:
        # Unet + MLP with convLSTM (Total params: 4,170,923)
        #unet_model = get_unet_mlp_convlstm(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE,[200,100,50], opt, class_weight)
        # Small Unet + MLP with convLSTM (Total params: 1,220,267)
        #unet_model = get_unetsmall_mlp_convlstm(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE,[200,100,50], opt, class_weight)
        # Unet + MLP TimeDistributed, and then ConvLSTM2D
        #unet_model = get_unet_mlp_td_convlstm(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE,[200,100,50], opt, class_weight)
        # All network with convLSTM (Total params: 4,176,312)
        unet_model = get_unet_mlp_convlstm_full(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE,[200,100,50], opt, class_weight)
        # Shallow MLP with LSTM
        #unet_model = get_sequential_mlp_LSTM(N_CHANNELS, N_TIMESTEPS, num_output_channels, PATCH_SIZE, PATCH_SIZE, t_mlp_shallow, opt, class_weight)
    if hvd.rank() == 0:
        print(unet_model.summary())
    #plot_model(unet_model, to_file=name_experiment + '/model.png', show_shapes=True, show_layer_names=True)
    json_string = unet_model.to_json()
    if hvd.rank() == 0:
        open(os.path.join(name_experiment, 'architecture.json'), 'w').write(json_string)  
        
    name_sub_experiment = os.path.basename(name_experiment)
    
    steps_per_epoch = int(NB_PATCH_PER_EPOCH / BATCH_SIZE / hvd.size())
    validation_steps = int(nb_valid_patch / BATCH_SIZE / hvd.size())
    print("Training on {0} batches, validating on {1} batches".format(NB_PATCH_PER_EPOCH, nb_valid_patch))

    if resume_training:
        unet_model.load_weights(snapshot_file_name)
    
    if not b_lstm:
        patch_estimates = DrawEstimates(hvd.rank())
        gen = data_generator(16, cnes_gen, 32, N_TIMESTEPS, use_background_layer, u_nb_tile, b_lstm)
        patches, gt_patches = next(gen)
        patch_estimates.set_data(patches, gt_patches, name_experiment=name_experiment)
        patch_estimates.draw_and_save()
    
    # callback kappa : batch_size=55 ==> max_message_size=2806087680  OverflowError: value too large to convert to int
    val_data = data_generator_without_mpi4py(int(nb_valid_patch / hvd.size()), cnes_gen, nb_valid_patch, N_TIMESTEPS, use_background_layer, b_lstm)
    x_val, y_val = next(val_data)
    o_kappa = clsvalidation_kappa(name_experiment, (x_val, y_val))


    tensorboard = TensorBoard(log_dir=os.path.join(name_experiment, 'logs_tensorboard'), write_graph=False)

    # save model if val_loss is better than before
    checkpointer = ModelCheckpoint(filepath=name_experiment+'/'+name_sub_experiment+'_best_weights.h5',
                                   verbose=2,
                                   monitor='val_loss',
                                   mode='auto',
                                   save_best_only=True)
    # save all models
    checkpointer_all = ModelCheckpoint(filepath=name_experiment+'/'+name_sub_experiment+'_weights.{epoch:02d}-{val_loss:.5f}.h5',
                                   verbose=2,
                                   monitor='val_loss',
                                   mode='auto',
                                   save_best_only=False)
    
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.00001, cooldown=5)

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, momentum_correction=True, steps_per_epoch=None, verbose=0)
    ]

    
    
    if hvd.rank() == 0:
        callbacks.append(checkpointer)
        callbacks.append(checkpointer_all)
        callbacks.append(tensorboard)
        #if not b_lstm:
            #callbacks.append(patch_estimates)

    if lr_reduce_auto == 1:
        callbacks.append(reduce_lr)
        
    callbacks.append(o_kappa)

    history = unet_model.fit_generator(data_generator(BATCH_SIZE, cnes_gen, NB_PATCH_PER_EPOCH, N_TIMESTEPS, use_background_layer, u_nb_tile, b_lstm),
                        steps_per_epoch=steps_per_epoch,
                        epochs=NUM_EPOCHS,
                        verbose=1,
                        validation_data=data_generator(BATCH_SIZE, cnes_gen, nb_valid_patch, N_TIMESTEPS, use_background_layer, u_nb_tile, b_lstm),
                        validation_steps=validation_steps,
                        callbacks=callbacks)
    

    hloss = np.asarray(history.history["loss"], np.float32)
#    print("np loss: ", hloss)
    
    if hloss.size < 5:
        mean_low_loss = np.mean(hloss)
    else:
        mean_low_loss = np.mean(hloss[-5:])

    all_loss = hvd.allgather([mean_low_loss]) 
                
    print("LOSS AVG: {} at rank {}".format(all_loss, hvd.rank()))
#    if hvd.rank() == 0:
#        print("all loss: ", all_loss)
        
    
#    print("mean loss:", mean_low_loss)
    return np.mean(all_loss)



if use_hyperas_optim:
    from hyperopt import hp    
    from hyperopt import Trials, STATUS_OK, tpe, fmin

    def train_model_wrapper(space):
        if hvd.rank() == 0:        
            #alpha =  #1000
            #beta = {{hpd.uniform(-0.2, -2)}} # -0.5
            #base_lr = {{hpd.loguniform(0.0001, 0.01)}} #0.001
            #lr_reduce_auto = {{hpd.choice([0, 1])}} #false
            hyperparams = [space['alpha'], space['beta'], space['base_lr'], space['lr_reduce_auto']]
        else:
            hyperparams = [0., 0., 0., 0.]

        all_hp = hvd.allgather(hyperparams)
        print("All hyperparams: {}".format(all_hp))
        if len(all_hp.shape) > 1:
            hps = all_hp[0,:]
        else:
            hps = all_hp
            
        alpha = hps[0]
        beta = hps[1]
        base_lr = hps[2]
        lr_reduce_auto = hps[3]

        return train_model(alpha, beta, base_lr, lr_reduce_auto)

    space = {'alpha' : hp.loguniform('alpha', np.log(1), np.log(1000)), 
                      'beta': hp.uniform('beta', -0.2, -2), 
                      'base_lr': hp.loguniform('base_lr', np.log(0.0001), np.log(0.01)),
                      'lr_reduce_auto' : hp.choice('lr_reduce_auto', [0, 1])}
        
    if hvd.rank() == 0:
        best_run, best_model = fmin(train_model_wrapper, space=space, algo=tpe.suggest, max_evals=20, trials=Trials())
    else:
        train_model_wrapper(None)
else:
    train_model(1000, -0.5, 0.0001, 1, prefix_sub_experiment=name_experiment)
    #train_model(1000, -0.5, 0.001, 1, prefix_sub_experiment=name_experiment)





if in_notebook:
    patch_estimates = DrawEstimates()
    gen = data_generator(32, cnes_gen, 32, N_TIMESTEPS, use_background_layer, b_lstm)
    patches, gt_patches = next(gen)
    preds_patches = unet_model.predict(patches,batch_size=4)
    patches = np.concatenate(patches, axis=1)
    patch_estimates.set_data(patches, gt_patches)
    patch_estimates.preds = preds_patches
    patch_estimates.draw_and_save()


# ======================
# Display sample results
# ======================
    import random
    from patch_display import make_mosaic_result, save_img, format_patches_for_display_colormap
    NB_SAMPLES = 15
    input_ch = [0,1,2] # input channels you want to visualize
    output_ch=[11,12] # output channels you want to visualize
    input_gain = 5 # if patch visualization is too dark, increase
    thresh_output = False # whether you want soft or hard output value

    # generate samples
    split_ds_idx = int(h5ds_num_data / 3.0 * 2.0)
    start_idx = random.randint(0,split_ds_idx)-NB_SAMPLES
    gen = data_generator(NB_SAMPLES, h5ds, start_idx, start_idx+NB_SAMPLES, min_channels, max_channels, use_background_layer, b_lstm)
    patches, gt_patches = next(gen)

    # get model prediction
    predictions = unet_model.predict(patches)
    print(gt_patches[0,1:10,:])
    print(predictions[0,1:10,:])
    # format result
    patches = np.concatenate(patches,axis=1)
    #print(patches.shape)
    plot_patch, gt_patches_viz, preds_patches_viz = format_patches_for_display_colormap(patches, gt_patches, predictions, 
                                        input_ch=input_ch, input_gain=5, colormap=color_map)
    #plot_patch, gt_patches_viz, preds_patches_viz = format_patches_for_display_colormap(patches, gt_patches, predictions, 
    #                                                                           input_ch=input_ch, output_ch=output_ch, 
    #                                                                           input_gain=input_gain, thresh_output=thresh_output) 
    mosaic = make_mosaic_result(plot_patch, gt_patches_viz, preds_patches_viz)

    # display  
    fig, ax = plt.subplots(figsize=(24, 14))
    if in_notebook:
        plt.imshow(mosaic)
        plt.show()
    else:
        save_img(make_mosaic_result(plot_patch, gt_patches_viz, preds_patches_viz),
                      name_experiment + '/eval_results')


