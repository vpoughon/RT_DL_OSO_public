# coding: utf-8

import os
import json



def read_reference_info(s_ref_info):
    '''
    reads info_references.txt in RasterData directory, to retrieve CLASS_ID_SET, label_map and color_map
    @param s_ref_info : info_references.txt in RasterData directory
    @return CLASS_ID_SET : list representing order in which labeled layers are organized
    @return label_map : map between class id and its name
    @return color map : map between class rank in CLASS_ID_SET and color to use to represent it in RGB
    '''
    if not os.path.isfile(s_ref_info):
        raise(Exception('Error: missing file {}'.format(s_ref_info)))
    fic = open(s_ref_info, 'r')
    t_lig = fic.readlines()
    fic.close()
    t_lig = [lig for lig in t_lig if not lig.startswith('#')]
    if len(t_lig) != 3:
        raise(Exception('Error: missing information in {}'.format(s_ref_info)))
    CLASS_ID_SET = json.loads(t_lig[0])
    label_map = json.loads(t_lig[1])
    label_map = {int(k):v for k, v in label_map.items()}
    color_map = json.loads(t_lig[2])
    color_map = {int(k):v for k, v in color_map.items()}
    return CLASS_ID_SET, label_map, color_map
    
