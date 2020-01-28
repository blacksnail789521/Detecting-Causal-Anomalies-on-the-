import platform
import pandas as pd
import json, csv
import pickle
from datetime import datetime
from operator import itemgetter
from pprint import pprint
import importlib
import itertools
from copy import deepcopy
import numpy as np
import math, time, collections, os, errno, sys, code, random
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import shutil
import configparser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import spatial



class RCAE2E:
    
    def __init__(self, \
                 \
                 data_folder_path, \
                 tool_id, \
                 normalize, \
                 sample, \
                 load_data_mode, \
                 \
                 r_w, \
                 t_w, \
                 K, \
                 _lambda, \
                 beta, \
                 alpha, \
                 maxIters, \
                 TICC_GTC_convergence_threshold, \
                 num_proc, \
                 output_path, \
                 \
                 c, \
                 tau, \
                 RCA_CTC_convergence_threshold, \
                 \
                 significant_difference_threshold):
        
        self.data_folder_path = data_folder_path
        self.tool_id = tool_id
        self.normalize = normalize
        self.sample = sample
        self.load_data_mode = load_data_mode
        
        self.r_w = r_w
        self.t_w = t_w
        self.K = K
        self._lambda = _lambda
        self.beta = beta
        self.alpha = alpha
        self.maxIters = maxIters
        self.TICC_GTC_convergence_threshold = TICC_GTC_convergence_threshold
        self.num_proc = num_proc
        self.output_path = output_path
        
        self.TICC_GTC_mode = None
        self.r = None
        self.t = None
        self.T = None
        
        self.c = c
        self.tau = tau
        self.RCA_CTC_convergence_threshold = RCA_CTC_convergence_threshold
        
        self.significant_difference_threshold = significant_difference_threshold
    
    
    
    def fit(self):
    
        print("##########################################################################################################")
        print("##########################################################################################################")
        print("@@@@@@@@@@@@ RCAE2E start! @@@@@@@@@@@@")
        
        
        self.r = 2
                
        ### save as pickle
        
        with open(os.path.join(data_folder_path, "test", "anomalous_run_data_MRF" + ".pickle"), 'rb') as file:
            anomalous_run_data_MRF = pickle.load(file)
        
        with open(os.path.join(data_folder_path, "test", "ground_truth_run_data_MRF" + ".pickle"), 'rb') as file:
            ground_truth_run_data_MRF = pickle.load(file)
        
        
        ### RCA_CTC
        
        causal_anomaly_score = self.RCA_CTC(anomalous_run_data_MRF, ground_truth_run_data_MRF)
            
        ### save as pickle
                
        with open(os.path.join(data_folder_path, "test", "causal_anomaly_score" + ".pickle"), 'wb') as file:
            pickle.dump(causal_anomaly_score, file)
        
        print("@@@@@@@@@@@@ RCAE2E end! @@@@@@@@@@@@")
        print("##########################################################################################################")
        print("##########################################################################################################")
    
    
    
    def load_data(self):
        
        load_data_module = importlib.import_module("3-load_data")
        load_data_class = getattr(load_data_module, "load_data")
        load_data_instance = load_data_class(data_folder_path = self.data_folder_path, \
                                             tool_id = self.tool_id, \
                                             normalize = self.normalize, \
                                             sample = self.sample, \
                                             load_data_mode = self.load_data_mode)
        data = load_data_instance()
        
        return data
        
    
    
    def TICC_GTC(self, data):
        
        TICC_GTC_module = importlib.import_module("5-TICC_GTC")
        TICC_GTC_class = getattr(TICC_GTC_module, "TICC_GTC")
        TICC_GTC_instance = TICC_GTC_class(r_w = self.r_w, \
                                           t_w = self.t_w, \
                                           K = self.K, \
                                           _lambda = self._lambda, \
                                           beta = self.beta, \
                                           alpha = self.alpha, \
                                           maxIters = self.maxIters, \
                                           TICC_GTC_convergence_threshold = self.TICC_GTC_convergence_threshold,\
                                           num_proc = self.num_proc, \
                                           output_path = self.output_path, \
                                           \
                                           TICC_GTC_mode = self.TICC_GTC_mode, \
                                           r = self.r, \
                                           t = self.t, \
                                           T = self.T)
        
        return TICC_GTC_instance.fit(data = data)
        
    
    
    def compare_two_profiles(self, old_profile, new_profile):
        
        method = "test"
        
        if method == 0 or method == "test":
        
            difference_proportion_mean = []
            for idx in range( len(old_profile["cluster_assignment"]) ):
                old_MRF = np.asarray(old_profile["cluster_MRFs"][ old_profile["cluster_assignment"][idx] ])
                new_MRF = np.asarray(new_profile["cluster_MRFs"][ new_profile["cluster_assignment"][idx] ])
                difference = np.abs(old_MRF - new_MRF)
                #print("difference: " + str(difference))
                _sum = np.sum([old_MRF, new_MRF], axis = 0)
                #print("_sum: " + str(_sum))
                
                difference_proportion = []
                for i in range(len(old_MRF)):
                    difference_proportion.append([])
                    for j in range(len(old_MRF[0])):
                        if _sum[i][j] != 0:
                            difference_proportion[i].append(difference[i][j] / _sum[i][j])
                        else:
                            difference_proportion[i].append(0)
                difference_proportion = np.asarray(difference_proportion)
                
                #difference_proportion = np.divide(difference, _sum, where = _sum != 0)
                #print("difference_proportion: " + str(difference_proportion))
                difference_proportion_mean.append(np.mean(difference_proportion))
            ### drop all nan (ya why not)
            #print("difference_proportion_mean: " + str(difference_proportion_mean))
            #print(~np.isnan(difference_proportion_mean))
            #difference_proportion_mean = difference_proportion_mean[~np.isnan(difference_proportion_mean)]
            difference_proportion_mean_all_MRFs = np.mean(difference_proportion_mean)
            if method != "test":
                print("difference_proportion_mean_all_MRFs: " + str(difference_proportion_mean_all_MRFs))
            
            profiles_difference = difference_proportion_mean_all_MRFs
            profiles_difference_method_0 = profiles_difference
        
        
        if method == 1 or method == "test":
        
            difference_proportion = []
            for idx in range( len(old_profile["cluster_assignment"]) ):
            # for idx in range(1):
                old_MRF = np.asarray(old_profile["cluster_MRFs"][ old_profile["cluster_assignment"][idx] ])
                new_MRF = np.asarray(new_profile["cluster_MRFs"][ new_profile["cluster_assignment"][idx] ])
                #print("old_MRF: " + str(old_MRF))
                #print("new_MRF: " + str(new_MRF))
                difference = np.abs(old_MRF - new_MRF)
                difference = np.sum(difference)
                #print("difference: " + str(difference))
                _sum = np.sum(old_MRF) + np.sum(new_MRF)
                #print("_sum: " + str(_sum))
                
                difference_proportion.append(difference / _sum)
                
            #print("difference_proportion: " + str(difference_proportion))

            difference_proportion_mean = np.mean(difference_proportion)
            if method != "test":
                print("difference_proportion_mean: " + str(difference_proportion_mean))
            
            profiles_difference = difference_proportion_mean
            profiles_difference_method_1 = profiles_difference
            
            
        if method == 2 or method == "test":
            
            cosine_distance = []
            for idx in range( len(old_profile["cluster_assignment"]) ):
            # for idx in range(1):
                old_MRF = old_profile["cluster_MRFs"][ old_profile["cluster_assignment"][idx] ]
                new_MRF = new_profile["cluster_MRFs"][ new_profile["cluster_assignment"][idx] ]
                cosine_distance.append(spatial.distance.cosine( list(itertools.chain(*old_MRF)), list(itertools.chain(*new_MRF)) ))
                
            #print("difference_proportion: " + str(difference_proportion))

            cosine_distance_mean = np.mean(cosine_distance)
            if method != "test":
                print("cosine_distance_mean: " + str(cosine_distance_mean))
            
            profiles_difference = cosine_distance_mean
            profiles_difference_method_2 = profiles_difference
        
        
        if method == "test":
            print("####################################################################")
            print("profiles_difference_method_0: " + str(profiles_difference_method_0))
            print("profiles_difference_method_1: " + str(profiles_difference_method_1))
            print("profiles_difference_method_2: " + str(profiles_difference_method_2))
            print("####################################################################")
        
        
        if profiles_difference >= self.significant_difference_threshold:
            significant_difference = True
        else:
            significant_difference = False
        
        return significant_difference
        
        
    
    def RCA_CTC(self, anomalous_run_data_MRF, ground_truth_run_data_MRF):
        
        RCA_CTC_module = importlib.import_module("6-RCA_CTC")
        RCA_CTC_class = getattr(RCA_CTC_module, "RCA_CTC")
        RCA_CTC_instance = RCA_CTC_class(c = self.c, \
                                         tau = self.tau, \
                                         RCA_CTC_convergence_threshold = self.RCA_CTC_convergence_threshold, \
                                         \
                                         t_w = self.t_w, \
                                         r = self.r)
        
        s = RCA_CTC_instance.fit(anomalous_run_data_MRF = anomalous_run_data_MRF, \
                                 ground_truth_run_data_MRF = ground_truth_run_data_MRF)
        
        return s
    
        
    
if __name__ == "__main__":
    
    ### get parameters form parameters.ini
    
    config = configparser.ConfigParser()
    config.read("parameters.ini")
    
    
    ### parameters for load data
    
    data_folder_path = eval(config.get("load_data", "data_folder_path"))
    tool_id = config.get("load_data", "tool_id")
    normalize = config.getboolean("load_data", "normalize")
    sample = config.getint("load_data", "sample")
    load_data_mode = config.getint("load_data", "load_data_mode")
    
    
    ### parameters for TICC_GTC
    
    r_w = config.getint("TICC_GTC", "r_w")
    t_w = config.getint("TICC_GTC", "t_w")
    K = config.getint("TICC_GTC", "K")
    _lambda = config.getfloat("TICC_GTC", "_lambda")
    beta = config.getfloat("TICC_GTC", "beta")
    alpha = config.getfloat("TICC_GTC", "alpha")
    maxIters = config.getint("TICC_GTC", "maxIters")
    TICC_GTC_convergence_threshold = config.getfloat("TICC_GTC", "TICC_GTC_convergence_threshold")
    num_proc = config.getint("TICC_GTC", "num_proc")
    output_path = eval(config.get("TICC_GTC", "output_path"))
    
    
    ### parameters for RCA_CTC
    
    c = config.getfloat("RCA_CTC", "c")
    tau = config.getfloat("RCA_CTC", "tau")
    RCA_CTC_convergence_threshold = config.getfloat("RCA_CTC", "RCA_CTC_convergence_threshold")
    
    
    ### parameters for RCAE2E
    
    significant_difference_threshold = config.getfloat("RCAE2E", "significant_difference_threshold")
    
    
    ### call RCAE2E
    
    RCAE2E_instance = RCAE2E(data_folder_path = data_folder_path, \
                             tool_id = tool_id, \
                             normalize = normalize, \
                             sample = sample, \
                             load_data_mode = load_data_mode, \
                             \
                             r_w = r_w, \
                             t_w = t_w, \
                             K = K, \
                             _lambda = _lambda, \
                             beta = beta, \
                             alpha = alpha, \
                             maxIters = maxIters, \
                             TICC_GTC_convergence_threshold = TICC_GTC_convergence_threshold,\
                             num_proc = num_proc, \
                             output_path = output_path, \
                             \
                             c = c, \
                             tau = tau, \
                             RCA_CTC_convergence_threshold = RCA_CTC_convergence_threshold, \
                             \
                             significant_difference_threshold = significant_difference_threshold)
    RCAE2E_instance.fit()