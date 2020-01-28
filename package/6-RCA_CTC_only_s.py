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




class RCA_CTC:

    def __init__(self, \
                 \
                 c, \
                 tau, \
                 RCA_CTC_convergence_threshold, \
                 \
                 t_w, \
                 r):

        self.c = c
        self.tau = tau
        self.RCA_CTC_convergence_threshold = RCA_CTC_convergence_threshold
        
        self.t_w = t_w
        self.r = r
        self.N = None
        self.T = None
    
    
    
    def fit(self, \
            anomalous_run_data_MRF, \
            ground_truth_run_data_MRF):
            
        ### A: anomalous_run_data_MRF
        
        A = np.asarray(anomalous_run_data_MRF)
        
        
        ### G: ground_truth_run_data_MRF
        
        G = np.asarray(ground_truth_run_data_MRF)
        
        
        ### B: anomalous_run_data_broken_MRF
        
        B = G - A
        
        
        ### s: causal_anomaly_score
        
        s = []
        
        
        ### calculate N and T
        
        self.N = int(len(A[0]) / self.t_w)
        self.T = len(A) + self.t_w - 1
        
        
        ### calculate all causal anomaly score
        
        for t in range(self.t_w - 1, self.T):
            print("-------------------------------------------------------")
            print("t: " + str(t))
            
            
            if t == self.t_w - 1:
                ### calculate_causal_anomaly_score (input: t_w zeros, B[t], G[t])
                s.append(self.calculate_causal_anomaly_score([ [random.random() for _ in range(self.N)] for _ in range(self.t_w) ], \
                                                             deepcopy(B[(t) - self.t_w + 1]), \
                                                             deepcopy(G[(t) - self.t_w + 1])))
            else:
                ### calculate_causal_anomaly_score (input: previous s, B[t], G[t])
                ### if all elements in previous s are zeros, input random instead.
                s.append(self.calculate_causal_anomaly_score([ [random.random() for _ in range(self.N)] for _ in range(self.t_w) ] \
                                                             if all(element == 0 for element in list(itertools.chain(*deepcopy(s[t - self.t_w])))) \
                                                             else deepcopy(s[t - self.t_w]), \
                                                             deepcopy(B[(t) - self.t_w + 1]), \
                                                             deepcopy(G[(t) - self.t_w + 1])))
        print("-------------------------------------------------------")
        
        
        ### average_all_causal_anomaly_score
        
        s = self.average_all_causal_anomaly_score(s)
        
        
        ### draw_s
        
        self.draw_s(s)
        
        return s
    
    
    
    def calculate_causal_anomaly_score(self, s, B, G):
    
        ### delete the first element and copy the last element as the last element
        
        s.pop(0)
        s.append(s[len(s) - 1])
        #print("s (initail): " + str(s))
        
        
        ### convert s to single vector (reverse at first!)
        
        s = np.asarray(list(itertools.chain(*list(reversed(s)))))
        
        ### calculate ~B and ~G
        
        print("B: \n" + str(B))
        B = self.calculate_degree_normalized_matrix(B)
        print("~B: \n" + str(B))
        
        print("G: \n" + str(G))
        G = self.calculate_degree_normalized_matrix(G)
        print("~G: \n" + str(G))
        
        
        ### calculate E and M
        
        E, M = self.calculate_E_and_M(G)
        
        
        ### update s until RCA_CTC_convergence_threshold
        
        new_s = s
        new_objective_function_value = float('Inf')
        
        while True:
        
            old_s = new_s
            old_objective_function_value = new_objective_function_value
            
            new_s, new_objective_function_value = self.update_s_and_objective_function_value(old_s, B, G, E, M)
            print("####################")
            print("old_objective_function_value: " + str(old_objective_function_value))
            print("new_objective_function_value: " + str(new_objective_function_value))
            print("####################")
            if abs(old_objective_function_value - new_objective_function_value) < self.RCA_CTC_convergence_threshold:
                #print("RCA_CTC_convergence_threshold!")
                s = new_s
                break
        
        
        ### convert s to multiple vectors (reverse at last!)
        
        multiple_s = []
        for t in range(self.t_w):
            multiple_s.append([])
            for n in range(self.N):
                multiple_s[t].append(s[t * self.N + n])
        s = list(reversed(multiple_s))
        
        #print("s (final): " + str(s))
        
        return s
    
    
    
    def calculate_degree_normalized_matrix(self, matrix):
    
        ### calculate degree matrix
        
        D = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                D[i][i] += matrix[i][j]
        D = np.power(D, -0.5, where = D != 0)
        
        return np.matmul(np.matmul(D, matrix), D)
        
    
    
    def calculate_E_and_M(self, G):
        
        ### calculate E
        
        E = (1 - self.c) * np.linalg.inv(np.identity(self.N * self.t_w) - self.c * G)
        print("E: \n" + str(E))
        
        
        ### calculate M
        
        M = []
        for i in range(len(G)):
            M.append([])
            for j in range(len(G[i])):
                if G[i][j] != 0:
                    M[i].append(1)
                else:
                    M[i].append(G[i][j])
        M = np.asarray(M)
        #print("M: \n" + str(M))
        
        return E, M
    
    
    
    def update_s_and_objective_function_value(self, old_s, B, G, E, M):
        
        ### calculate new_s
        
        numerator = np.matmul(4 * np.multiply(np.matmul(np.transpose(E), \
                                                        B), \
                                              M), \
                              np.matmul(E, \
                                        old_s))
                             
        #print("numerator: " + str(numerator))
        denominator = np.matmul(4 * np.multiply(np.matmul(np.matmul(np.transpose(E), \
                                                                    np.matmul(E, \
                                                                              old_s)), \
                                                          np.matmul(np.transpose(old_s), \
                                                                    np.transpose(E))), \
                                                M), \
                                np.matmul(E, \
                                          old_s)) \
                    + self.tau * np.ones(self.N * self.t_w)
        #print("denominator: " + str(denominator))
        
        new_s = np.multiply(old_s, \
                            np.power(np.divide(numerator, \
                                               denominator, \
                                               where = denominator != 0), \
                                     0.25))
                       
                       
        ### calculate new_objective_function_value
        
        TMP = np.matmul(np.matmul(E, \
                                  new_s), \
                        np.matmul(np.transpose(new_s), \
                                  np.transpose(E)) \
                        )
        TMP = np.multiply(TMP, M)
        TMP = np.subtract(TMP, B)
        
        new_objective_function_value = np.power(np.linalg.norm(TMP), 2) + self.tau * np.linalg.norm(new_s, ord = 1)
        
        print("old_s: " + str(np.isnan(old_s).any()))
        print("new_s: " + str(np.isnan(new_s).any()))
        if np.isnan(new_s).any():
            print("new_s: " + str(new_s))
            print("numerator: " + str(numerator))
            print("denominator: " + str(denominator))
            print("ok: " + str(np.divide(numerator, \
                                               denominator, \
                                               where = denominator != 0)))
            print("ok_power: " + str(np.power(np.divide(numerator, \
                                               denominator, \
                                               where = denominator != 0), \
                                     0.25)))
        print("B: " + str(np.isnan(B).any()))
        print("G: " + str(np.isnan(G).any()))
        print("E: " + str(np.isnan(E).any()))
        print("M: " + str(np.isnan(M).any()))
        
        
        
        return new_s, new_objective_function_value
    
    
    
    def average_all_causal_anomaly_score(self, s):
        
        average_s = []
        for t in range(self.T):
            average_s.append([])
            for i in range(self.T - self.t_w + 1):
                for j in range(self.t_w):
                    if i + j == t:
                        average_s[t].append(s[i][j])
        #print(average_s)
        
        for t in range(self.T):
            average_s[t] = list(map(lambda items: float(sum(items)) / len(items), zip(*average_s[t])))
        #print(average_s)
        
        return average_s
    
    
    def draw_s(self, s):
        
        ### calculate transpose_s
        
        transpose_s = list(map(list, zip(*s)))
        
        
        ### draw
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 10))
        img = ax.imshow(transpose_s, cmap='gray_r')
        ax.set_yticks([i for i in range(len(transpose_s))])
        ax.tick_params(labelsize=15)
        ax.set_xlabel("time", fontsize = 15)
        ax.set_ylabel("sensor", fontsize = 15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_title("r = " + str(self.r), fontsize = 24)
        fig.colorbar(img, cax=cax)
        plt.tight_layout(h_pad=1)
        
        
        ### calculate output_path_for_one_para_comb and create folder
        
        config = configparser.ConfigParser()
        config.read("parameters.ini")
        
        data_folder_path = eval(config.get("load_data", "data_folder_path"))
        tool_id = config.get("load_data", "tool_id")
        output_path = eval(config.get("TICC_GTC", "output_path"))
        _lambda = config.getfloat("TICC_GTC", "_lambda")
        K = config.getint("TICC_GTC", "K")
        beta = config.getfloat("TICC_GTC", "beta")
        alpha = config.getfloat("TICC_GTC", "alpha")
        r_w = config.getint("TICC_GTC", "r_w")
        
        output_path_for_one_para_comb = os.path.join(output_path, \
                                                     "lambda=" + str(_lambda) + \
                                                     " K=" + str(K) + \
                                                     " beta=" + str(beta) + \
                                                     " alpha=" + str(alpha) + \
                                                     " t_w=" + str(self.t_w) + \
                                                     " r_w=" + str(r_w), \
                                                     \
                                                     "RCA_CTC", \
                                                     \
                                                     "r=" + str(r) + \
                                                     " (run " + str(r) + ")")
        print(output_path_for_one_para_comb)
                                                     
        if not os.path.exists(output_path_for_one_para_comb):
            try:
                os.makedirs(output_path_for_one_para_comb)
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise
        
        
        ### save figure
        
        fig.savefig(os.path.join(output_path_for_one_para_comb, "causal_anomaly_score.jpg"))
        plt.close("all")
    
        
        
if __name__ == "__main__":
    
    ### get parameters form parameters.ini
    
    config = configparser.ConfigParser()
    config.read("parameters.ini")
    
    
    ### parameters for RCA_CTC
    
    c = config.getfloat("RCA_CTC", "c")
    tau = config.getfloat("RCA_CTC", "tau")
    RCA_CTC_convergence_threshold = config.getfloat("RCA_CTC", "RCA_CTC_convergence_threshold")
    
    t_w = config.getint("TICC_GTC", "t_w")
    r = config.getint("RCA_CTC_test", "r")
    
    
    ### test data
    
    # A: anomalous_run_data_MRF
    A = []
    A.append([[0, 1, 0, 0, 0, 0, 0, 0 ,0], \
              [1, 0, 1, 0, 0, 1, 0, 0, 0], \
              [0, 1, 0, 0, 0, 0, 1, 0, 0], \
              [0, 0, 0, 0, 1, 0, 0, 0, 0], \
              [0, 0, 0, 1, 0, 1, 0, 0, 1], \
              [0, 1, 0, 0, 1, 0, 0, 0, 0], \
              [0, 0, 1, 0, 0, 0, 0, 1, 0], \
              [0, 0, 0, 0, 0, 0, 1, 0, 1], \
              [0, 0, 0, 0, 1, 0, 0, 1, 0]])
    #     ______________
    #    /              \
    #   o      o      o  \
    #  /      /      /   |
    # o---o  o---o  o---o
    #      \/     \/
    
    A.append([[0, 0, 0, 0, 0, 0, 0, 0 ,0], \
              [0, 0, 1, 0, 0, 1, 0, 0, 0], \
              [0, 1, 0, 0, 0, 0, 1, 0, 0], \
              [0, 0, 0, 0, 0, 0, 0, 0, 0], \
              [0, 0, 0, 0, 0, 1, 0, 0, 1], \
              [0, 1, 0, 0, 1, 0, 0, 0, 0], \
              [0, 0, 1, 0, 0, 0, 0, 0, 0], \
              [0, 0, 0, 0, 0, 0, 0, 0, 1], \
              [0, 0, 0, 0, 1, 0, 0, 1, 0]])
    #     ______________
    #    /              \
    #   o      o      o  \
    #                    |
    # o---o  o---o  o---o
    #      \/     \/
    
    A.append([[0, 0, 0, 0, 0, 0, 0, 0 ,0], \
              [0, 0, 1, 0, 0, 1, 0, 0, 0], \
              [0, 1, 0, 0, 0, 0, 0, 0, 0], \
              [0, 0, 0, 0, 0, 0, 0, 0, 0], \
              [0, 0, 0, 0, 0, 1, 0, 0, 1], \
              [0, 1, 0, 0, 1, 0, 0, 0, 0], \
              [0, 0, 0, 0, 0, 0, 0, 0, 0], \
              [0, 0, 0, 0, 0, 0, 0, 0, 1], \
              [0, 0, 0, 0, 1, 0, 0, 1, 0]])
    #      
    #                    
    #   o      o      o   
    #  /      /      /    
    # o---o  o---o  o---o
    #      \/     \/
    
    A = np.asarray(A)

    # G: ground_truth_run_data_MRF
    ground_truth_run_data_MRF = [[0, 1, 0, 0, 0, 0, 0, 0 ,0], \
                                 [1, 0, 1, 0, 0, 1, 0, 0, 0], \
                                 [0, 1, 0, 0, 0, 0, 1, 0, 0], \
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0], \
                                 [0, 0, 0, 1, 0, 1, 0, 0, 1], \
                                 [0, 1, 0, 0, 1, 0, 0, 0, 0], \
                                 [0, 0, 1, 0, 0, 0, 0, 1, 0], \
                                 [0, 0, 0, 0, 0, 0, 1, 0, 1], \
                                 [0, 0, 0, 0, 1, 0, 0, 1, 0]]
    #     ______________
    #    /              \
    #   o      o      o  \
    #  /      /      /   |
    # o---o  o---o  o---o
    #      \/     \/
    
    G = []
    G.append(ground_truth_run_data_MRF)
    G.append(ground_truth_run_data_MRF)
    G.append(ground_truth_run_data_MRF)
    G = np.asarray(G)
    
    
    ### call RCA_CTC
    
    RCA_CTC_instance = RCA_CTC(c = c, \
                               tau = tau, \
                               RCA_CTC_convergence_threshold = RCA_CTC_convergence_threshold, \
                               \
                               t_w = t_w, \
                               r = r)
    s = RCA_CTC_instance.fit(anomalous_run_data_MRF = A, \
                             ground_truth_run_data_MRF = G)