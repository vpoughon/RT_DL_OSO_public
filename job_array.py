#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

"""
Module de gestion des job array

@version: 1.0

@author: Vincent Poulain (THALES)
@date: decembre 2015
"""
import os
import datetime
import subprocess
import time

class jobArray:
    '''
    
    '''
    def __init__(self, s_name, s_work_dir, CLUSTER_CHECK_TIME = 30, b_wait=1):
        '''
        initialise les fichiers necessaires au job array (params, script, sortie qsub) :
        s_jobArrayParamFile : nom du fichier de parametres
        s_jobArrayScript  : nom du script a executer
        s_rep_qsub_out : nom du repertoire qui contient les sorties stdout et stderr
        @param s_name : nom identifiant le job array.
        @params_work_dir : repertoire de travail
        @param b_wait : 1 pour attendre la fin de l'execution du job array
        '''
        self.s_name = s_name
        self.s_work_dir = s_work_dir
        # suffixe pour les fichiers copies dans le repertoire de sortie (pour differencier les runs realises dans le meme repertoire)
        self.suffixe = datetime.datetime.now().isoformat().split('T')[1].replace(':','').replace('.','') # correspond a l'heure actuelle : 13:32:16.896847 devient 133216896847
        #   repertoire contenant les fichiers de sortie de qsub
        self.s_rep_qsub_out = os.path.join(self.s_work_dir, 'sorties_qsub_'+s_name+'_'+self.suffixe)+'/'
        if not os.path.isdir(self.s_rep_qsub_out):
            #returnCode = os.system("mkdir -p " + self.s_rep_qsub_out)
            os.mkdir(self.s_rep_qsub_out)
        #   creation et execution d'un script JobArray pour paralleliser :
        self.s_jobArrayParamFile = os.path.join(self.s_work_dir, 'JobArrayParamFile_'+s_name+'_'+self.suffixe+'.dat')
        self.s_jobArrayScript = os.path.join(self.s_work_dir, 'JobArrayScript_'+s_name+'_'+self.suffixe+'.sh')
        self.job_array_id = None
        self.CLUSTER_CHECK_TIME = CLUSTER_CHECK_TIME
        self.b_wait = b_wait
    
    def execute(self):
        '''
        execute le job array
        '''
        p = subprocess.Popen(["qsub "+self.s_jobArrayScript], shell=True, stdout=subprocess.PIPE)
        value = p.communicate()[0].decode("utf-8")
        if value == '':
            return 1
        self.job_array_id = value.replace('\n','').split('.')[0]
        print("job_array_id: {0}".format(self.job_array_id))
        #print "Execution du traitement pour chaque produit..."
        if self.b_wait:
            # on attend que tous les produits soient traites :
            self.waitCluster() # on attend qu'il n'y ait plus de jobs en attente
        # modification des droits sur les fichier .ER et .OU contenant les sorties standards et erreur des jobs :
        os.system("chmod -R 750 "+self.s_rep_qsub_out)
        return 0
    
    def getId(self):
        return self.job_array_id

    def getParamFile(self):
        return self.s_jobArrayParamFile
    
    def getScriptFile(self):
        return self.s_jobArrayScript
    
    def getQsubRepOut(self):
        return self.s_rep_qsub_out
    
    def getSuffixe(self):
        return self.suffixe
    
    # pour savoir quand l'execution du job array est terminee :
    def waitCluster(self):
        _s_time = self.CLUSTER_CHECK_TIME
        while True:
            p = subprocess.Popen(["qstat -xp "+self.job_array_id], shell=True, stdout=subprocess.PIPE)
            value = p.communicate()
            status = value[0].split()[-2].decode("utf-8") # retourne le status du job array : Q pour en queue, B pour en cours, F pour termine
            #u_count = int(value[0].replace('\n',''))
            if status == 'F':
                return
            else:
                p_status = subprocess.Popen(["qstat -xt "+self.job_array_id], shell=True, stdout=subprocess.PIPE)
                value_status = p_status.communicate()[0].decode("utf-8")
                value_status = ' '.join(value_status.split('\n')[3:-2])
                print("Job state, waiting/running/finished : {0}/{1}/{2}".format(\
                         value_status.count(' Q '), value_status.count(' R '), \
                         value_status.count(' X ') + value_status.count(' F ')))
                time.sleep(_s_time) # on attend quelques secondes

