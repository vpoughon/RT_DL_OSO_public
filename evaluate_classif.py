'''
Evaluation of classification map:
    - computes contingency and confusion matrices with OTB
    - write a xls file with results

Example :
python evaluate_classif.py -img /work/OT/siaa/Work/RTDLOSO/scripts/release_06022018/sentinel2_mlp_weights_T31TDN_4noeuds_instance1_30b_11t_batch16_adamlr0_0001_weightpatch_test29/sentinel2_mlp_weights_T31TDN_4noeuds_instance1_30b_11t_batch16_adamlr0_0001_weightpatch_test29_classif.tif -label /work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/ReferenceData_by_tile/T31TDN/testing.shp -out /work/OT/siaa/Work/RTDLOSO/scripts/release_06022018/sentinel2_mlp_weights_T31TDN_4noeuds_instance1_30b_11t_batch16_adamlr0_0001_weightpatch_test29


'''
import os
import sys
import xlsxwriter
import numpy as np
import argparse
import subprocess

def evaluate(s_segmentation_map, s_test_gt_shp_file, s_out_dir, colormap_file):
    os.system('mkdir -p {0}'.format(s_out_dir))
    s_tile = os.path.basename(os.path.dirname(s_test_gt_shp_file))
    s_contingency_matrix = os.path.join(s_out_dir, 'contmat.csv')
    # generate contingency matrix
    t_cmd = [". /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_whl.sh && otbcli_ComputeConfusionMatrix", 
                    "-in", s_segmentation_map,
                    "-out", s_contingency_matrix,
                    "-ref", "vector",
                    "-ref.vector.in", s_test_gt_shp_file,
                    "-ref.vector.field","CODE2",
                    "-format", "contingencytable"]
    s_cmd = ' '.join(t_cmd)
    process = subprocess.Popen([s_cmd], shell=True,
                              stdout=subprocess.PIPE)
    process.wait()
    #print(s_cmd)
    
    
    # generate confusion matrix
    s_confusion_matrix = os.path.join(s_out_dir, 'confmat.csv')
    t_cmd = [". /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_whl.sh && otbcli_ComputeConfusionMatrix", 
                    "-in", s_segmentation_map,
                    "-out", s_confusion_matrix,
                    "-ref", "vector",
                    "-ref.vector.in", s_test_gt_shp_file,
                    "-ref.vector.field","CODE2"]
    s_cmd = ' '.join(t_cmd)
    process = subprocess.Popen([s_cmd], shell=True,
                              stdout=subprocess.PIPE)
    
    
    
    
    # write logfile
    ct = 0
    logfile = open(os.path.join(s_out_dir, 'logfile'), 'w')
    for line in iter(process.stdout.readline, ''):
        #sys.stdout.write(line)
        sys.stdout.write(line.decode('utf-8'))
        ct += 1
        if ct > 1000:
            break
        logfile.write(line.decode('utf-8')) # remove decode for python 2.7
    process.wait()
    logfile.close()
    
    # Ajout VP
    label_map = { 32: "Foret pers.", 
              34: "Pelouses",
              36: "Lande lign.",
              211: "Prairie",
              41: "Bat. denses",
              10: "Cult annuelles",
              11: "Cult ete",
              12: "Cult hiver",
              43: "ZI",
              44: "Routes",
              221: "Verger",
              51: "Eau",
              42: "Bat. diffus",
              222: "Vigne",
              31: "Foret caduques",
              45: "Surf. minerales",
              46: "Plages/dunes",
              53: "Glaciers/neige"}
    # get Kappa and OA from logfile:
    o_log = open(os.path.join(s_out_dir, 'logfile'), 'r')
    s_log = o_log.read()
    o_log.close()
    s_kappa = s_log[s_log.find('Kappa index: ') + len('Kappa index: ') : s_log.find('\n', s_log.find('Kappa index: '))]
    print(s_kappa)
    s_OA = s_log[s_log.find('Overall accuracy index: ') + len('Overall accuracy index: ') : s_log.find('\n', s_log.find('Overall accuracy index: '))]
    print(s_OA)
    # create xls file from contingency matrix
    t_contmat = np.loadtxt(s_contingency_matrix, delimiter=',', dtype='str')
    # attention, il est possible que certaines classes soient absentes de l'axe x (si pas dans la verite terrain), on les ajoute pour ne pas fausser le reste
    t_all_classes = list(set([val for val in t_contmat[1::,0]] + [val for val in t_contmat[0,1::]]))
    t_all_classes = sorted([int(val) for val in t_all_classes])
    t_all_classes = [str(val) for val in t_all_classes]
    #print(t_all_classes)
    t_class_manquantes1 = [val for val in t_contmat[:,0] if val not in t_contmat[0,:]]
    print(t_class_manquantes1)
    for s_class in t_class_manquantes1:
        u_pos = min(t_all_classes.index(s_class) + 1, len(t_contmat[0, :]))
        t_contmat = np.insert(t_contmat, u_pos, [0], axis=1)
        t_contmat[0, u_pos] = s_class
        #print(t_contmat[0,:])
    # et si des classes  sont absentes de y (si jamais predites), on les ajoute dans la matrice
    #elif t_contmat.shape[0] < t_contmat.shape[1]:
    t_class_manquantes2 = [val for val in t_contmat[0,:] if val not in t_contmat[:,0]]
    print(t_class_manquantes2)
    for s_class in t_class_manquantes2:
        #u_pos = list(t_contmat[0,:]).index(s_class)
        u_pos = min(t_all_classes.index(s_class) + 1, len(t_contmat[:,0]))
        t_contmat = np.insert(t_contmat, u_pos, [0], axis=0)
        t_contmat[u_pos, 0] = s_class
    s_xlsname = os.path.join(s_out_dir, 'contmat_{}.xls'.format(s_tile))
    t_contmat[0,0] = '0'
    t_contmat = t_contmat.astype(int)
    t_sum = np.sum(t_contmat, axis=0, keepdims=True)
    t_percentmat = np.around(t_contmat[1:, 1:] * 100 / t_sum[:, 1:], decimals=1)
    t_surface_ha_mat = np.around(t_contmat[1:, 1:] / 100, decimals=0)
    print(t_contmat.shape)
    print(t_contmat)
    # precision, recall
    #t_prec = np.around(np.array([t_contmat[i, i] * 100 / t_sum[0, i] for i in range(1, t_contmat.shape[1])]), decimals=2)
    #t_recall = np.around(np.array([t_contmat[i, i] * 100 / np.sum(t_contmat, axis=1, keepdims=True)[i, 0] for i in range(1, t_contmat.shape[1])]), decimals=2)
    # if true positive = 0 and false positive = 0 and false negative = 0, then precision=recall=1
    # if true positive = 0 and (false positive != 0 or false negative != 0), then precision=recall=0
    # cf https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    t_prec = np.around(np.array([t_contmat[i, i] * 100 / t_sum[0, i] \
                                 if t_sum[0, i] + np.sum(t_contmat, axis=1, keepdims=True)[i, 0] != 0 \
                                 else 100 \
                                 for i in range(1, t_contmat.shape[1])]), \
                                 decimals=2)
    t_recall = np.around(np.array([t_contmat[i, i] * 100 / np.sum(t_contmat, axis=1, keepdims=True)[i, 0] \
                                  if t_sum[0, i] + np.sum(t_contmat, axis=1, keepdims=True)[i, 0] != 0 \
                                  else 100 \
                                  for i in range(1, t_contmat.shape[1])]), \
                                  decimals=2)
    # Fscore
    t_prec_plus_recall = t_prec + t_recall
    t_fscore = np.around(np.divide(2 * t_recall * t_prec, t_prec_plus_recall, out=np.zeros_like(t_recall), where=t_prec_plus_recall!=0), decimals=2)
    # Global Fscore : weighted (according to the number of true instance of each label) and unweighted
    s_global_fscore_unweighted = str(np.mean(t_fscore))
    t_weights = [np.sum(t_contmat, axis=1, keepdims=True)[i, 0] for i in range(1, t_contmat.shape[1])]
    f_total_labels = np.sum(t_weights)
    t_weights = t_weights / f_total_labels
    s_global_fscore_weighted = str(np.sum([score * weigth for score, weigth in zip(t_fscore, t_weights)]))
    wb = xlsxwriter.Workbook(s_xlsname)
    ws = wb.add_worksheet('Confusion matrix')
    ws.conditional_format('B3:R4', {'type':'3_color_scale', 'min_value': 0.0, 'max_value': 100.0})
    ws.conditional_format('B8:R{}'.format(7+t_contmat.shape[0]), {'type':'2_color_scale', 'min_value': 0.0, 'max_value': 100.0, \
                          'min_color' : '#FFFFFF', 'max_color': '#008000'})
    ws.conditional_format('B5:R5', {'type':'3_color_scale', 'min_value': 0.0, 'max_value': 100.0})
    bold = wb.add_format({'bold': True})
    small = wb.add_format()
    small.set_font_size(8)
    # test name
    ws.write(0, 0, os.path.basename(s_out_dir), bold)
    # precision
    ws.write(2, 0, 'PREC', bold)
    [ws.write(2, i+1, t_prec[i]) for i in range(len(t_prec))]
    # recall
    ws.write(3, 0, 'RECALL', bold)
    [ws.write(3, i+1, t_recall[i]) for i in range(len(t_recall))]
    # FScore
    ws.write(4, 0, 'F-SCORE', bold)
    [ws.write(4, i+1, t_fscore[i]) for i in range(len(t_fscore))]
    # matrice de confusion avec pourcentages:
    ws.write(6, 0, 'LABELS', bold)
    for row in range(t_contmat.shape[0]):
        # labels
        if row > 0:
            ws.write(row + 6, 0, int(t_contmat[row, 0]), bold)
        for col in range(1, t_contmat.shape[1]):
            if row == 0:
                # labels
                ws.write(row + 5, col, label_map[int(t_contmat[row, col])], small)
                ws.write(row + 6, col, int(t_contmat[row, col]), bold)
            else:
                # percents
                ws.write(row + 6, col, float(t_percentmat[row-1, col-1]))
    # add Kappa and OA below
    ws.write(t_contmat.shape[0] + 6, 0, 'Kappa', bold)
    ws.write(t_contmat.shape[0] + 6, 1, s_kappa)
    ws.write(t_contmat.shape[0] + 6 + 1, 0, 'OA', bold)
    ws.write(t_contmat.shape[0] + 6 + 1, 1, s_OA)
    # add global Fscore weighted and unweighted
    ws.write(t_contmat.shape[0] + 6, 2, 'Fscore unweighted', bold)
    ws.write(t_contmat.shape[0] + 6, 4, s_global_fscore_unweighted)
    ws.write(t_contmat.shape[0] + 6 + 1, 2, 'Fscore weighted', bold)
    ws.write(t_contmat.shape[0] + 6 + 1, 4, s_global_fscore_weighted)
    # second sheet with number of pixels
    ws = wb.add_worksheet('Confusion matrix2')
    ws.conditional_format('B3:R4', {'type':'3_color_scale', 'min_value': 0.0, 'max_value': 100.0})
    #ws.conditional_format('B7:O20', {'type':'2_color_scale', 'min_value': 0.0, 'max_value': 100.0, \
                          #'min_color' : '#FFFFFF', 'max_color': '#008000'})
    ws.conditional_format('B5:R5', {'type':'3_color_scale', 'min_value': 0.0, 'max_value': 100.0})
    bold = wb.add_format({'bold': True})
    # test name
    ws.write(0, 0, os.path.basename(s_out_dir), bold)
    # precision
    ws.write(2, 0, 'PREC', bold)
    [ws.write(2, i+1, t_prec[i]) for i in range(len(t_prec))]
    # recall
    ws.write(3, 0, 'RECALL', bold)
    [ws.write(3, i+1, t_recall[i]) for i in range(len(t_recall))]
    # FScore
    ws.write(4, 0, 'F-SCORE', bold)
    [ws.write(4, i+1, t_fscore[i]) for i in range(len(t_fscore))]
    # matrice de confusion avec pourcentages:
    ws.write(6, 0, 'LABELS', bold)
    for row in range(t_contmat.shape[0]):
        # labels
        if row > 0:
            ws.write(row + 6, 0, int(t_contmat[row, 0]), bold)
        for col in range(1, t_contmat.shape[1]):
            if row == 0:
                # labels
                ws.write(row + 5, col, label_map[int(t_contmat[row, col])], small)
                ws.write(row + 6, col, int(t_contmat[row, col]), bold)
            else:
                # surface in ha
                ws.write(row + 6, col, float(t_surface_ha_mat[row-1, col-1]))
                
    # add Kappa and OA below
    ws.write(t_contmat.shape[0] + 6, 0, 'Kappa', bold)
    ws.write(t_contmat.shape[0] + 6, 1, s_kappa)
    ws.write(t_contmat.shape[0] + 6 + 1 , 0, 'OA', bold)
    ws.write(t_contmat.shape[0] + 6 + 1 , 1, s_OA)
    # add global Fscore weighted and unweighted
    ws.write(t_contmat.shape[0] + 6, 2, 'Fscore unweighted', bold)
    ws.write(t_contmat.shape[0] + 6, 4, s_global_fscore_unweighted)
    ws.write(t_contmat.shape[0] + 6 + 1, 2, 'Fscore weighted', bold)
    ws.write(t_contmat.shape[0] + 6 + 1, 4, s_global_fscore_weighted)
    wb.close()
    # fin Ajout VP
    
    
    # generate colored map
    t_cmd = ([". /work/OT/siaa/Work/RTDLOSO/init_env_python3.5_whl.sh &&  otbcli_ColorMapping", 
                    "-in", s_segmentation_map,
                    "-out", os.path.join(s_out_dir, 'color_map_{}.tif'.format(s_tile)),
                    "-method.custom.lut", colormap_file])
    s_cmd = ' '.join(t_cmd)
    #print(s_cmd)
    process = subprocess.run([s_cmd], shell=True, stdout=subprocess.PIPE)
    
    return 0

def main():
    '''
    main
    '''
    # Parser creation
    parser = argparse.ArgumentParser(description='Evaluate classification result')
    # Args
    parser.add_argument('-img', '--img', metavar='[SEGMENTATION_MAP]', help='Segmentation map to evaluate', required=True)
    parser.add_argument('-label', '--label', metavar='[TEST_LABEL]',\
                        help='testing.shp', required=True)
    parser.add_argument('-out', '--out', metavar='[OUT]', help='Output directory', required=True)
    parser.add_argument('-cmap', '--cmap', metavar='[COLOR_MAP]', help='Color map', default='/work/OT/siaa/Work/RTDLOSO/data/feat_S2_20152016_extract/support/color_map.txt', required=False)
    # Command line parsing
    args = vars(parser.parse_args())
    s_segmentation_map = os.path.abspath(args['img'])
    s_test_gt_shp_file = os.path.abspath(args['label'])
    s_out_dir = os.path.abspath(args['out'])
    colormap_file = os.path.abspath(args['cmap'])
    
    evaluate(s_segmentation_map, s_test_gt_shp_file, s_out_dir, colormap_file)
    

if __name__ == '__main__':
    main()
    
