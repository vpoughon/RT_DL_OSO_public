# -*- coding: utf-8 -*-

'''
R&T generation de cartes d'occupation des sols par reseaux de neurones convolutionnels

Launch training with mpi

python train_mpi.py -rep /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract -tile 'T31UDQ T31TDN' -raster /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/RasterData_RPG2016_UA -out sentinel2_mlp_weights_T31UDQ_T31TDN_8noeuds_instance1_30b_11t_batch32_adamlr0_0001_test49 -recover false -nodes 8 -epochs 50
'''
import os
import subprocess
import argparse


def create_job_and_execute(s_rep_img, s_idtile, s_raster, s_out, s_recover, nb_nodes=8, u_epochs=75):
    nb_procs = 24
    nb_mpiprocs = 1
    #walltime="150:00:00" # for LSTM 
    walltime="72:00:00"
    memory = "120000mb"
    s_script_qsub = os.path.join(s_out, 'script_mpi.sh')
    s_qsub_log_dir = os.path.join(s_out, "qsub")
    os.system('mkdir -p {0}'.format(s_qsub_log_dir))
    # launch qsub command
    s_qsub_content = '\n'.join(['#!/bin/bash',\
                        '#PBS -N DeepLearning_MPI',\
                        '#PBS -l select={0}:ncpus={1}:mpiprocs={2}:mem={3}'.format(nb_nodes, nb_procs, nb_mpiprocs, memory),\
                        '#PBS -l walltime={0}'.format(walltime),\
                        '#PBS -e {0}'.format(s_qsub_log_dir),\
                        '#PBS -o {0}'.format(s_qsub_log_dir),\
                        'NB_PROC=$(wc -l \"${PBS_NODEFILE}\" | cut -d\" \" -f 1)',\
                        #'source /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_MKL.sh',\
                        'source /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_MKL_update.sh',\
                        #'source /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_whl.sh',\
                        #'source /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_stack_ml.sh',\
                        'cd ${PBS_O_WORKDIR}',\
                        'echo NB_PROC:',\
                        'echo $NB_PROC',\
                        #'export HOROVOD_TIMELINE={0}'.format(os.path.join(s_out, 'timeline.json')),\
                        #'export OMP_NUM_THREAD=24',\
                        #'export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0',\
                        #'export CNES_DBG_NO_TILE_LOAD=1',\
                        #'export CNES_DBG_BATCH_SHARING=1',\
                        #'mpirun -x HOROVOD_TIMELINE -np $NB_PROC --hostfile $PBS_NODEFILE \
                        'mpirun -np $NB_PROC --hostfile $PBS_NODEFILE \
                        python /work/OT/siaa/Work/RTDLOSO/scripts/multi_tuile_mpi4py/Sentinel2_train_demo_hvd.py \
                        -rep {0} -tile \'{1}\' -raster {2} -out {3} -recover {4} -epochs {5}'.format(s_rep_img, s_idtile, s_raster, s_out, s_recover, u_epochs)])
    
    
    o_script = open(s_script_qsub, 'w')
    o_script.write(s_qsub_content)
    o_script.close()
    os.system('chmod +x {0}'.format(s_script_qsub))
    p = subprocess.Popen(['qsub {0}'.format(s_script_qsub)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #p = subprocess.Popen(['qsub -a 201807280500 {0}'.format(s_script_qsub)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # with the starting date
    stdout, stderr = p.communicate()
    stdout = stdout.decode("utf-8")
    last_id = stdout.strip('\n')
    print ("%s" % str(last_id))


def main():
    '''
    main
    '''
    # Parser creation
    parser = argparse.ArgumentParser(description='Run learning with MPI')
    # Args
    parser.add_argument('-rep', '--rep', metavar='[REP_IMAGE]', help='', required=True)
    parser.add_argument('-tile', '--tile', metavar='[ID_TILE]', help='Tile ID', required=True)
    parser.add_argument('-raster', '--raster', metavar='[RASTER_DATA]', help='Directory containing rasterized labeled data', required=True)
    parser.add_argument('-out', '--out', metavar='[OUT_DIR]',\
                        help='Output directory that will contain the learned model', required=True)
    parser.add_argument('-recover', '--recover', metavar='[RECOVER]', help='true/false to  allow to start training from a saved model', required=True)
    parser.add_argument('-nodes', '--nodes', metavar='[NODE_NUMBER]', help='Number of cluster nodes to use', default=8, required=False)
    parser.add_argument('-epochs', '--epochs', metavar='[EPOCH_NUMBER]', help='Number of epochs', default=75, required=False)
    # Command line parsing
    args = vars(parser.parse_args())
    s_rep_img = os.path.abspath(args['rep'])
    s_idtile = args['tile']
    s_recover = args['recover']
    s_raster_dir = args['raster']
    s_out = args['out']
    u_nodes = int(args['nodes'])
    u_epochs = int(args['epochs'])
    os.system('mkdir -p {0}'.format(s_out))
    if s_recover not in ['true', 'false']:
        raise Exception('--recover must be true or false')
    create_job_and_execute(s_rep_img, s_idtile, s_raster_dir, s_out, s_recover, u_nodes, u_epochs)
    return 0

if __name__ == '__main__':
    main()
