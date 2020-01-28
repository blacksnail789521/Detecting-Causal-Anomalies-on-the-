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
from snap import *



##Parameters to play with

t_w = 3
N = 10
T_for_single_cluster = 200
_lambda = 0.2
K = 3
R = 5
T = T_for_single_cluster * K

seg_ids = []
for r in range(R):
    for k in range(K):
        seg_ids.append(k)

break_points = []
for i in range(len(seg_ids)):
    break_points.append(i+1)
break_points = np.asarray(break_points) * T_for_single_cluster

rand_seed = 10
save_inverse_covarainces = True
###########################################################




block_matrices = {} ##Stores all the block matrices

def generate_inverse(rand_seed):
	np.random.seed(rand_seed)
	def genInvCov(size, low = 0.3 , upper = 0.6, portion = 0.2,symmetric = True):
		portion = portion/2
		S = np.zeros((size,size))
		# low = abs(low)
		# upper = abs(upper)
		G = GenRndGnm(PNGraph, size, int((size*(size-1))*portion))
		for EI in G.Edges():
			value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0]) 
			# print value
			S[EI.GetSrcNId(), EI.GetDstNId()] = value
		if symmetric:
			S = S + S.T
		# vals = alg.eigvalsh(S)
		# S = S + (0.1 - vals[0])*np.identity(size)
		return np.matrix(S)

	def genRandInv(size,low = 0.3, upper=0.6, portion = 0.2):
		S = np.zeros((size,size))
		for i in range(size):
			for j in range(size):
				if np.random.rand() < portion:
					value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0]) 
					S[i,j] = value
		return np.matrix(S)

	##Generate all the blocks
	for block in range(t_w):
		if block ==0:
			block_matrices[block] = genInvCov(size = N, portion = _lambda, symmetric = (block == 0) )
		else:
			block_matrices[block] = genRandInv(size = N, portion = _lambda)

	##Initialize the inverse matrix
	inv_matrix = np.zeros([t_w*N,t_w*N])

	##go through all the blocks
	for block_i in range(t_w):
		for block_j in range(t_w):
			block_num = np.abs(block_i - block_j)
			if block_i > block_j:
				inv_matrix[block_i*N:(block_i+1)*N, block_j*N:(block_j+1)*N] = block_matrices[block_num]
			else:
				inv_matrix[block_i*N:(block_i+1)*N, block_j*N:(block_j+1)*N] = np.transpose(block_matrices[block_num])

	##print out all the eigenvalues
	eigs, _ = np.linalg.eig(inv_matrix)
	lambda_min = min(eigs)

	##Make the matrix positive definite
	inv_matrix = inv_matrix + (0.1 + abs(lambda_min))*np.identity(N*t_w)

	eigs, _ = np.linalg.eig(inv_matrix)
	lambda_min = min(eigs)

	##Save the matrix to file
	# np.savetxt("matrix_random_seed=" + str(rand_seed) + ".csv", inv_matrix, delimiter =",",fmt='%1.2f')
	return inv_matrix


    
    
    
    
    

    
### prepare output directory
    
tool_id = "Synthetic_Data"

if platform.system() == "Windows":
    save_data_path = r"DATA\input"
else:
    save_data_path = r"data/input"
save_data_path = os.path.join(save_data_path, tool_id)

if not os.path.exists(save_data_path):
    try:
        os.makedirs(save_data_path)
    except OSError as exc:  # Guard against race condition of path already existing
        if exc.errno != errno.EEXIST:
            raise
    
    
    
    
    
    
    
############GENERATE POINTS
cluster_mean = np.zeros([N,1])
cluster_mean_stacked = np.zeros([N*t_w,1])

##Generate two inverse matrices
cluster_inverses = {}
cluster_covariances = {}
for cluster in range(K):
    cluster_inverses[cluster] = generate_inverse(rand_seed = cluster)
    cluster_covariances[cluster] = np.linalg.inv(cluster_inverses[cluster])
    if save_inverse_covarainces:
        np.savetxt(os.path.join(save_data_path, "Inverse Covariance cluster = " + str(cluster) + ".csv"), \
                   cluster_inverses[cluster], delimiter= ",", fmt='%1.6f')
        np.savetxt(os.path.join(save_data_path, "Covariance cluster = "+ str(cluster) +".csv"), \
                   cluster_covariances[cluster], delimiter= ",", fmt='%1.6f')


##data matrix
data = np.zeros([break_points[-1],N])
data_stacked = np.zeros([break_points[-1]-t_w+1, N*t_w])
cluster_point_list = []
for counter in range(len(break_points)):
	break_pt = break_points[counter]
	cluster = seg_ids[counter]
	if counter == 0:
		old_break_pt = 0
	else:
		old_break_pt = break_points[counter-1]
	for num in range(old_break_pt,break_pt):
		##generate the point from this cluster
		if num == 0:
			cov_matrix = cluster_covariances[cluster][0:N,0:N]##the actual covariance matrix
			new_mean = cluster_mean_stacked[N*(t_w-1):N*t_w]
			##Generate data
			new_row = np.random.multivariate_normal(new_mean.reshape(N),cov_matrix)
			data[num,:] = new_row

		elif num < t_w:
			##The first section
			cov_matrix = cluster_covariances[cluster][0:(num+1)*N,0:(num+1)*N] ##the actual covariance matrix
			n = N
			Sig22 = cov_matrix[(num)*n:(num+1)*n,(num)*n:(num+1)*n] 
			Sig11 = cov_matrix[0:(num)*n,0:(num)*n]
			Sig21 = cov_matrix[(num)*n:(num+1)*n,0:(num)*n]
			Sig12 = np.transpose(Sig21)
			cov_mat_tom = Sig22 - np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),Sig12) #sigma2|1
			log_det_cov_tom = np.log(np.linalg.det(cov_mat_tom))# log(det(sigma2|1))
			inv_cov_mat_tom = np.linalg.inv(cov_mat_tom)# The inverse of sigma2|1

			##Generate data
			a = np.zeros([(num)*N,1])
			for idx in range(num):
				a[idx*N:(idx+1)*N,0] = data[idx,: ].reshape([N])
			new_mean = cluster_mean + np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),(a - cluster_mean_stacked[0:(num)*N,:]) )
			new_row = np.random.multivariate_normal(new_mean.reshape(N),cov_mat_tom)
			data[num,:] = new_row

		else:
			cov_matrix = cluster_covariances[cluster]##the actual covariance matrix
			n = N
			Sig22 = cov_matrix[(t_w-1)*n:(t_w)*n,(t_w-1)*n:(t_w)*n] 
			Sig11 = cov_matrix[0:(t_w-1)*n,0:(t_w-1)*n]
			Sig21 = cov_matrix[(t_w-1)*n:(t_w)*n,0:(t_w-1)*n]
			Sig12 = np.transpose(Sig21)
			cov_mat_tom = Sig22 - np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),Sig12) #sigma2|1
			log_det_cov_tom = np.log(np.linalg.det(cov_mat_tom))# log(det(sigma2|1))
			inv_cov_mat_tom = np.linalg.inv(cov_mat_tom)# The inverse of sigma2|1

			a = np.zeros([(t_w-1)*N,1])
			for idx in range(t_w-1):
				a[idx*N:(idx+1)*N,0] = data[num - t_w + 1 + idx,: ].reshape([N])

			new_mean = cluster_mean + np.dot(np.dot(Sig21,np.linalg.inv(Sig11)),(a - cluster_mean_stacked[0:(t_w-1)*N,:]) )

			new_row = np.random.multivariate_normal(new_mean.reshape(N),cov_mat_tom)
			data[num,:] = new_row

data = data.tolist()
print("done with generating the data!!!")
print("length of generated data is:", len(data))







##save the generated matrix
header_string = "STATE, "
for i in range(N):
    if i != N-1:
        header_string += "N" + str(i) + ", "
    else:
        header_string += "N" + str(i)

cluster_assignment = [element for element in seg_ids for _ in range(T_for_single_cluster)]

for element_idx, element in enumerate(data):
    element.insert(0, cluster_assignment[element_idx])

np.savetxt(os.path.join(save_data_path, tool_id + "_original" + ".csv"), \
           data, \
           delimiter = ",", \
           fmt = "%i," + "%1.4f," * (N - 1) + "%1.4f", \
           header = header_string, \
           comments = "")



attribute = ["STATE"]
for i in range(N):
    attribute.append("N" + str(i))

data_index = []
for r in range(R):
    data_index.append({"LOT_ID": "R" + str(r), \
                       "length": T, \
                       "start_index": r * T, \
                       "end_index": (r + 1) * T - 1})
    
    
    
    
### save data_index + attribute to json file

with open(os.path.join(save_data_path, tool_id + "_original" + ".json"), "w") as output_file:
    json.dump(obj = {"data_index": data_index, \
                     "attribute": attribute}, \
              fp = output_file, \
              indent = 4)



              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              
              

