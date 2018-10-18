# -*- coding: utf-8 -*-

'''
R&T generation de cartes d'occupation des sols par reseaux de neurones convolutionnels

Launch Random Forest training with qsub


python RF_qsub.py -rep /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract -tile 'T30TXQ T31TGK T31UDQ' -out /work/OT/siaa/Work/RTDLOSO/tests/RF/MultiTuiles/
'''
import os
import subprocess
import argparse


def create_job_and_execute(s_rep_img, s_idtile, s_out, s_vector):
    nb_nodes = 1
    nb_procs = 24
    walltime="72:00:00"
    memory = "120000mb"
    s_script_qsub = os.path.join(s_out, 'script_qsub.sh')
    s_qsub_log_dir = os.path.join(s_out, "qsub")
    os.system('mkdir -p {0}'.format(s_qsub_log_dir))
    # construct otb cli command:
    s_img = ''
    s_training = ''
    s_valid = ''
    for s_tile in s_idtile.split(' '):
        s_img += os.path.join(s_rep_img, s_tile, 'Sentinel2_ST_GAPFIL.tif') + ' '
        s_training += os.path.join(s_vector, s_tile, 'training.shp') + ' '
        s_valid += os.path.join(s_vector, s_tile, 'testing.shp') + ' '
    s_cmd = 'OTB_LOGGER_LEVEL=DEBUG otbcli_TrainImagesClassifier -io.il '+ s_img + ' -io.vd ' + s_training + ' -io.valid ' +  s_valid + ' -io.out ' +  \
            os.path.join(s_out, 'model_' + '_'.join(s_idtile.split(' ')) + '.rf') + \
            ' -sample.vfn CODE2 -classifier rf -classifier.rf.nbtrees 100 -classifier.rf.max 20 -classifier.rf.cat 17 -ram 10000\n'
    # launch qsub command
    s_qsub_content = '\n'.join(['#!/bin/bash',\
                        '#PBS -N RF',\
                        '#PBS -l select={}:ncpus={}:mem={}'.format(nb_nodes, nb_procs, memory),\
                        '#PBS -l walltime={0}'.format(walltime),\
                        '#PBS -e {0}'.format(s_qsub_log_dir),\
                        '#PBS -o {0}'.format(s_qsub_log_dir),\
                        'source /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_MKL_OTB6.4.sh',\
                        'cd ${PBS_O_WORKDIR}',\
                        s_cmd])
    
    
    o_script = open(s_script_qsub, 'w')
    o_script.write(s_qsub_content)
    o_script.close()
    os.system('chmod +x {0}'.format(s_script_qsub))
    p = subprocess.Popen(['qsub {0}'.format(s_script_qsub)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    parser.add_argument('-vector', '--vector', metavar='[VECTOR_DATA]', help='Directory containing vector reference data by tile', \
                        default="/work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile",  required=False)
    parser.add_argument('-out', '--out', metavar='[OUT_DIR]',\
                        help='Output directory that will contain the learned model', required=True)
    # Command line parsing
    args = vars(parser.parse_args())
    s_rep_img = os.path.abspath(args['rep'])
    s_idtile = args['tile']
    s_vector = args['vector']
    s_out = os.path.join(os.path.abspath(args['out']), '_'.join(s_idtile.split(' ')))
    os.system('mkdir -p {0}'.format(s_out))
    create_job_and_execute(s_rep_img, s_idtile, s_out, s_vector)
    return 0

if __name__ == '__main__':
    main()
