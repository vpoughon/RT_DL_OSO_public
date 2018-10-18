# -*- coding: utf-8 -*-

'''
R&T generation de cartes d'occupation des sols par reseaux de neurones convolutionnels

Launch Random Forest classification with qsub


python RF_classifier_qsub.py -rep /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract -model /work/OT/siaa/Work/RTDLOSO/tests/RF/MultiTuiles/T30TXQ_T31TGK_T31UDQ/model_T30TXQ_T31TGK_T31UDQ.rf -tiles 'T30TXQ T31TGK T31UDQ T31TDN T30TWT' -out /work/OT/siaa/Work/RTDLOSO/tests/RF/MultiTuiles/T30TXQ_T31TGK_T31UDQ -raster /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/RasterData_RPG2016_UA        -vector /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_RPG2016_UA_by_tile
'''
import os
import subprocess
import argparse


def create_job_and_execute(s_rep_img, s_model, t_idtile, s_raster_path, s_vector_path, s_out):
    nb_nodes = 1
    nb_procs = 10
    walltime="10:00:00"
    memory = "50000mb"
    s_script_qsub = os.path.join(s_out, 'script_qsub_classify.sh')
    s_qsub_log_dir = os.path.join(s_out, "qsub")
    os.system('mkdir -p {0}'.format(s_qsub_log_dir))
    # construct otb cli command:
    s_cmd = ''
    for s_tile in t_idtile:
        s_img = os.path.join(s_rep_img, s_tile, 'Sentinel2_ST_GAPFIL.tif')
        s_file_out = os.path.join(s_out, s_tile, 'classif_{}.tif'.format(s_tile))
        s_file_reg_out = os.path.join(s_out, s_tile, 'classif_{}_reg.tif'.format(s_tile))
        s_cmd += 'OTB_LOGGER_LEVEL=DEBUG otbcli_ImageClassifier -in ' + s_img +  \
                 ' -out ' + s_file_out + ' uint8 -model ' + s_model + '\n'
        s_cmd += 'OTB_LOGGER_LEVEL=DEBUG otbcli_ClassificationMapRegularization -ip.radius 1 -ip.suvbool 0 -io.in ' + s_file_out + ' -io.out ' + s_file_reg_out + ' uint8 \n'
        s_cmd += 'python ../evaluate_classif.py -img ' + s_file_reg_out + ' -label ' + os.path.join(s_vector_path, s_tile, 'testing.shp ') + ' -out ' + os.path.dirname(s_file_out) + \
        ' -cmap ' + os.path.join(s_raster_path, 'color_map.txt') + '\n'
    # launch qsub command
    s_qsub_content = '\n'.join(['#!/bin/bash',\
                        '#PBS -N RF',\
                        '#PBS -l select={}:ncpus={}:mem={}'.format(nb_nodes, nb_procs, memory),\
                        '#PBS -l walltime={0}'.format(walltime),\
                        '#PBS -e {0}'.format(s_qsub_log_dir),\
                        '#PBS -o {0}'.format(s_qsub_log_dir),\
                        'source /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_MKL_OTBdev.sh',\
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
    parser.add_argument('-rep', '--rep', metavar='[REP_IMAGE]', help='', default='/work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract', required=False)
    parser.add_argument('-model', '--model', metavar='[MODEL]', help='', required=True)
    parser.add_argument('-tiles', '--tiles', metavar='[ID_TILES]', help='Tile ID to classify, separated by a space', required=True)
    parser.add_argument('-raster', '--raster', metavar='[RASTER_DATA]', help='Directory containing rasterized labeled data', required=True)
    parser.add_argument('-vector', '--vector', metavar='[VECTOR_DATA]', help='Directory containing vector reference data by tile', required=True)
    parser.add_argument('-out', '--out', metavar='[OUT_DIR]',\
                        help='Output directory that will contain a rep for each classified tile', required=True)
    # Command line parsing
    args = vars(parser.parse_args())
    s_rep_img = os.path.abspath(args['rep'])
    s_model = os.path.abspath(args['model'])
    t_idtile = args['tiles'].split(' ')
    s_raster_path = args['raster']
    s_vector_path = args['vector']
    s_out = os.path.abspath(args['out'])
    for s_id_tile in t_idtile:
        os.system('mkdir -p {0}'.format(os.path.join(s_out, s_id_tile)))
    create_job_and_execute(s_rep_img, s_model, t_idtile, s_raster_path, s_vector_path, s_out)
    return 0

if __name__ == '__main__':
    main()
