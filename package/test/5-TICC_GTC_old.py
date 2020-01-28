import platform
import pandas as pd
import os
import json
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
import pandas as pd
from multiprocessing import Pool
import shutil
import configparser



class ADMMSolver:
    def __init__(self, lamb, num_stacked, size_blocks, rho, S, rho_update_func=None):
        self.lamb = lamb
        self.numBlocks = num_stacked
        self.sizeBlocks = size_blocks
        probSize = num_stacked*size_blocks
        self.length = int(probSize*(probSize+1)/2)
        self.x = np.zeros(self.length)
        self.z = np.zeros(self.length)
        self.u = np.zeros(self.length)
        self.rho = float(rho)
        self.S = S
        self.status = 'initialized'
        self.rho_update_func = rho_update_func

    def ij2symmetric(self, i,j,size):
        return (size * (size + 1))/2 - (size-i)*((size - i + 1))/2 + j - i

    def upper2Full(self, a):
        n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)  
        A = np.zeros([n,n])
        A[np.triu_indices(n)] = a 
        temp = A.diagonal()
        A = (A + A.T) - np.diag(temp)             
        return A 

    def Prox_logdet(self, S, A, eta):
        d, q = np.linalg.eigh(eta*A-S)
        q = np.matrix(q)
        X_var = ( 1/(2*float(eta)) )*q*( np.diag(d + np.sqrt(np.square(d) + (4*eta)*np.ones(d.shape))) )*q.T
        x_var = X_var[np.triu_indices(S.shape[1])] # extract upper triangular part as update variable      
        return np.matrix(x_var).T

    def ADMM_x(self):    
        a = self.z-self.u
        A = self.upper2Full(a)
        eta = self.rho
        x_update = self.Prox_logdet(self.S, A, eta)
        self.x = np.array(x_update).T.reshape(-1)

    def ADMM_z(self, index_penalty = 1):
        a = self.x + self.u
        probSize = self.numBlocks*self.sizeBlocks
        z_update = np.zeros(self.length)

        # TODO: can we parallelize these?
        for i in range(self.numBlocks):
            elems = self.numBlocks if i==0 else (2*self.numBlocks - 2*i)/2 # i=0 is diagonal
            for j in range(self.sizeBlocks):
                startPoint = j if i==0 else 0
                for k in range(startPoint, self.sizeBlocks):
                    locList = [((l+i)*self.sizeBlocks + j, l*self.sizeBlocks+k) for l in range(int(elems))]
                    if i == 0:
                        lamSum = sum(self.lamb[loc1, loc2] for (loc1, loc2) in locList)
                        indices = [self.ij2symmetric(loc1, loc2, probSize) for (loc1, loc2) in locList]
                    else:
                        lamSum = sum(self.lamb[loc2, loc1] for (loc1, loc2) in locList)
                        indices = [self.ij2symmetric(loc2, loc1, probSize) for (loc1, loc2) in locList]
                    pointSum = sum(a[int(index)] for index in indices)
                    rhoPointSum = self.rho * pointSum

                    #Calculate soft threshold
                    ans = 0
                    #If answer is positive
                    if rhoPointSum > lamSum:
                        ans = max((rhoPointSum - lamSum)/(self.rho*elems),0)
                    elif rhoPointSum < -1*lamSum:
                        ans = min((rhoPointSum + lamSum)/(self.rho*elems),0)

                    for index in indices:
                        z_update[int(index)] = ans
        self.z = z_update

    def ADMM_u(self):
        u_update = self.u + self.x - self.z
        self.u = u_update

    # Returns True if convergence criteria have been satisfied
    # eps_abs = eps_rel = 0.01
    # r = x - z
    # s = rho * (z - z_old)
    # e_pri = sqrt(length) * e_abs + e_rel * max(||x||, ||z||)
    # e_dual = sqrt(length) * e_abs + e_rel * ||rho * u||
    # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
    # Returns (boolean shouldStop, primal residual value, primal threshold,
    #          dual residual value, dual threshold)
    def CheckConvergence(self, z_old, e_abs, e_rel, verbose):
        norm = np.linalg.norm
        r = self.x - self.z
        s = self.rho * (self.z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = math.sqrt(self.length) * e_abs + e_rel * max(norm(self.x), norm(self.z)) + .0001
        e_dual = math.sqrt(self.length) * e_abs + e_rel * norm(self.rho * self.u) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Debugging information to print(convergence criteria values)
            print('  r:', res_pri)
            print('  e_pri:', e_pri)
            print('  s:', res_dual)
            print('  e_dual:', e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    #solve
    def __call__(self, maxIters, eps_abs, eps_rel, verbose):
        num_iterations = 0
        self.status = 'Incomplete: max iterations reached'
        for i in range(maxIters):
            z_old = np.copy(self.z)
            self.ADMM_x()
            self.ADMM_z()
            self.ADMM_u()
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(z_old, eps_abs, eps_rel, verbose)
                if stop:
                    self.status = 'Optimal'
                    break
                new_rho = self.rho
                if self.rho_update_func:
                    new_rho = rho_update_func(self.rho, res_pri, e_pri, res_dual, e_dual)
                scale = self.rho / new_rho
                rho = new_rho
                self.u = scale*self.u
            if verbose:
                # Debugging information prints current iteration #
                print('Iteration %d' % i)
        return self.x







class TICC_GTC:
    def __init__(self, \
                 mode, \
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
                 r, \
                 t, \
                 T, \
                 ground_truth_r, \
                 compute_BIC = False, \
                 cluster_reassignment = 20):
        """
        Parameters:
            - t_w: size of the sliding window
            - r_w: size of the run window
            - K: number of clusters
            - _lambda: sparsity parameter
            - beta: local temporal consistency parameter
            - alpha: global temporal consistency parameter
            - maxIters: number of iterations
            - TICC_GTC_convergence_threshold: convergence threshold
            - output_path: output directory if necessary
            - cluster_reassignment: number of points to reassign to a 0 cluster
        """
        self.mode = mode
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
        self.r = r
        self.t = t
        self.T = T
        self.ground_truth_r = ground_truth_r
        
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.t_w + 1
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

    def fit(self, data):
        """
        Main method for TICC_GTC solver.
        Parameters:
            - data: data file
        """
        assert self.maxIters > 0  # must have at least one iteration
        
        
        # The basic folder to be created
        
        output_path_for_one_para_comb = self.prepare_out_directory()
        
        
        ### Get data into proper format (show parameter first!)
        
        if self.mode == 0:
            
            self.log_parameters()
            
            complete_D_train, training_indices, num_train_points, N, time_series_row_size = self.get_data_into_proper_format(data)
            
        else:
            
            self.r_w = 1
            self.K = 1
            self.beta = 0
            self.alpha = 0
            
            complete_D_train, training_indices, num_train_points, N, time_series_row_size = self.get_data_into_proper_format(data)
        
        
        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.K, covariance_type="full")
        gmm.fit(complete_D_train)
        clustered_points = gmm.predict(complete_D_train)
        gmm_clustered_pts = clustered_points + 0
        # K-means
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(complete_D_train)
        clustered_points_kmeans = kmeans.labels_  # todo, is there a difference between these two?
        kmeans_clustered_pts = kmeans.labels_

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc)  # multi-threading
        for iters in range(self.maxIters):
            if self.mode == 0:
                print("\nITERATION ###", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.K)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, N, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance
            
            if self.mode == 0:
                print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                 'computed_covariance': computed_covariance,
                                 'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                 'complete_D_train': complete_D_train,
                                 'N': N}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(list) # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.K)}

            before_empty_cluster_assign = clustered_points.copy()



            if iters != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.K, i]), i) for i in
                                 range(self.K)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.K):
                    if len_new_train_clusters[cluster_num] == 0:
                        cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num, "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.K, cluster_num] = old_computed_covariance[
                                self.K, cluster_selected]
                            cluster_mean_stacked_info[self.K, cluster_num] = complete_D_train[
                                                                                              point_to_move, :]
                            cluster_mean_info[self.K, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                  (self.t_w - 1) * N:self.t_w * N]

            for cluster_num in range(self.K):
                if self.mode == 0:
                    print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))
            
            if self.mode == 0:
                self.draw_cluster_assignments(output_path_for_one_para_comb, clustered_points, training_indices, iters)

            # TEST SETS STUFF
            # LLE + swtiching_penalty
            # Segment length
            # Create the F1 score from the graphs from k-means and GMM
            # Get the train and test points
            train_confusion_matrix_EM = self.compute_confusion_matrix(self.K, clustered_points,
                                                                 training_indices)
            train_confusion_matrix_GMM = self.compute_confusion_matrix(self.K, gmm_clustered_pts,
                                                                  training_indices)
            train_confusion_matrix_kmeans = self.compute_confusion_matrix(self.K, kmeans_clustered_pts,
                                                                     training_indices)
            ###compute the matchings
            matching_EM, matching_GMM, matching_Kmeans = self.compute_matches(train_confusion_matrix_EM,
                                                                              train_confusion_matrix_GMM,
                                                                              train_confusion_matrix_kmeans)
            
            if self.mode == 0:
                if np.array_equal(old_clustered_points, clustered_points):
                    print("\nCONVERGED!!! BREAKING EARLY!!!")
                    break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        if pool is not None:
            pool.close()
            pool.join()
        train_confusion_matrix_EM = self.compute_confusion_matrix(self.K, clustered_points,
                                                             training_indices)
        train_confusion_matrix_GMM = self.compute_confusion_matrix(self.K, gmm_clustered_pts,
                                                              training_indices)
        train_confusion_matrix_kmeans = self.compute_confusion_matrix(self.K, clustered_points_kmeans,
                                                                 training_indices)

        self.compute_f_score(matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                             train_confusion_matrix_GMM, train_confusion_matrix_kmeans)

        if self.compute_BIC:
            bic = self.computeBIC(self.K, time_series_row_size, clustered_points, train_cluster_inverse,
                             empirical_covariances)
            return clustered_points, train_cluster_inverse, bic
        
        profile = self.draw_MRF_and_write_profile(output_path_for_one_para_comb, clustered_points, train_cluster_inverse, N)
        return profile
    
    def get_data_into_proper_format(self, data):
        
        if self.mode == 0:
            
            N = len(data[0][0])
            time_series_row_size = (self.T - self.t_w + 1) * self.r_w
            print("length(data points): " + str(self.T))
            print("length(data subsequences): " + str(self.T - self.t_w + 1))
            print("N: " + str(N))
            print("time_series_row_size: " + str(time_series_row_size))
            num_train_points = time_series_row_size
            training_indices = np.arange(num_train_points)
            
            #print("data: " + str(data))
            complete_D_train = []
            
            # Stack the training data (change data points to subsequences)
            for r in range(self.r_w):
                for t in range(self.T - self.t_w + 1):
                    # change [[1, 2, 3], [4, 5, 6]] to [1, 2, 3, 4, 5, 6]
                    complete_D_train.append( list(itertools.chain(*data[r][t : t + self.t_w])) )
            
            complete_D_train = np.asarray(complete_D_train)
            #print("complete_D_train: " + str(complete_D_train))
            
            return complete_D_train, training_indices, num_train_points, N, time_series_row_size
        
        else:
            
            N = len(data[0])
            time_series_row_size = (self.T - self.t_w + 1) * self.r_w
            #print("length(data points): " + str(self.T))
            #print("length(data subsequences): " + str(self.T - self.t_w + 1))
            #print("N: " + str(N))
            #print("time_series_row_size: " + str(time_series_row_size))
            num_train_points = time_series_row_size
            training_indices = np.arange(num_train_points)
            
            #print("data: " + str(data))
            complete_D_train = []
            
            # Stack the training data (change data points to subsequences)
            for t in range(self.T - self.t_w + 1):
                # change [[1, 2, 3], [4, 5, 6]] to [1, 2, 3, 4, 5, 6]
                complete_D_train.append( list(itertools.chain(*data)) )
            
            complete_D_train = np.asarray(complete_D_train)
            #print("complete_D_train: " + str(complete_D_train))
            
            return complete_D_train, training_indices, num_train_points, N, time_series_row_size
    
    def plot_input_data(self, data):
        for run in data:
            print(len(run))
            plt.figure(figsize=(25, 8))
            plt.plot(run)
            plt.show()
        
    def compute_f_score(self, matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                        train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        f1_EM_tr = -1  # computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
        f1_GMM_tr = -1  # computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
        f1_kmeans_tr = -1  # computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)
        if self.mode == 0:
            print("\n\nTRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.K):
            matched_cluster__e_m = matching_EM[cluster]
            matched_cluster__g_m_m = matching_GMM[cluster]
            matched_cluster__k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster__e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster__g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster__k_means]

    def compute_matches(self, train_confusion_matrix_EM, train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        matching_Kmeans = self.find_matching(train_confusion_matrix_kmeans)
        matching_GMM = self.find_matching(train_confusion_matrix_GMM)
        matching_EM = self.find_matching(train_confusion_matrix_EM)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.K):
            matched_cluster_e_m = matching_EM[cluster]
            matched_cluster_g_m_m = matching_GMM[cluster]
            matched_cluster_k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster_e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster_g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster_k_means]
        return matching_EM, matching_GMM, matching_Kmeans

    def draw_cluster_assignments(self, output_path_for_one_para_comb, clustered_points, training_indices, iters):
    
        plt.figure()
        plt.plot(training_indices[0:len(clustered_points)], clustered_points, color="r")  # ,marker = ".",s =100)
        plt.ylim((-0.5, self.K + 0.5))
        plt.savefig(os.path.join(output_path_for_one_para_comb, "cluster assignments (iter=" + str(iters) + ").jpg"))
        plt.close("all")
    
    def draw_MRF_and_write_profile(self, output_path_for_one_para_comb, clustered_points, train_cluster_inverse, N):
    
        profile = {"cluster_assignment":[], "cluster_MRFs":{}}
        profile["cluster_assignment"] = clustered_points.tolist()
        for key, value in train_cluster_inverse.items():
            profile["cluster_MRFs"][key] = value.tolist()
        
        ### set MRF's all terms to positive values
        ### set MRF's all diagonal terms to zeros
        
        for key, value in profile["cluster_MRFs"].items():
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = abs(value[i][j])
                    if (i - j) % N == 0:
                        value[i][j] = 0
        
        
        ### plot every MRF
        
        for key, value in profile["cluster_MRFs"].items():
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 10))
            cax = ax.imshow(value, cmap='gray_r')
            ax.set_title(key)
            cbar = fig.colorbar(cax)
            if self.mode == 0:
                fig.savefig(os.path.join(output_path_for_one_para_comb, "MRF (cluster=" + str(key) + ").jpg"))
            else:
                fig.savefig(os.path.join(output_path_for_one_para_comb, \
                                         "MRF (time " + str(self.t - self.t_w + 1) + " to time " + str(self.t) + ").jpg"))
            plt.close("all")
        
        
        ### write profile
        
        if self.mode == 0:
            output_profile_name = "profile.json"
        else:
            output_profile_name = "t=" + str(self.t) + " (time " + str(self.t - self.t_w + 1) + " to time " + str(self.t) + ").json"
        with open(os.path.join(output_path_for_one_para_comb, output_profile_name), "w") as output_file:
            json.dump(obj = profile, fp = output_file, indent = 4)
        
        
        ### return profile
        
        if self.mode == 0:
            print("----------------------------------------")
            return profile
        else:
            return np.asarray(profile["cluster_MRFs"][0])

    def smoothen_clusters(self, cluster_mean_info, computed_covariance,
                          cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        for cluster in range(self.K):
            cov_matrix = computed_covariance[self.K, cluster][0:(self.num_blocks - 1) * n, \
                         0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        if self.mode == 0:
            print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([clustered_points_len, self.K])
        for point in range(clustered_points_len):
            if point + self.t_w - 1 < complete_D_train.shape[0]:
                for cluster in range(self.K):
                    cluster_mean = cluster_mean_info[self.K, cluster]
                    cluster_mean_stacked = cluster_mean_stacked_info[self.K, cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(self, computed_covariance, len_train_clusters, log_det_values, optRes, train_cluster_inverse):
        for cluster in range(self.K):
            if optRes[cluster] == None:
                continue
            val = optRes[cluster].get()
            if self.mode == 0:
                print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
            # THIS IS THE SOLUTION
            S_est = self.upperToFull(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.K, cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[self.K, cluster] = cov_out
            train_cluster_inverse[cluster] = X2
        for cluster in range(self.K):
            if self.mode == 0:
                print("length of the cluster ", cluster, "------>", len_train_clusters[cluster])

    def train_clusters(self, cluster_mean_info, cluster_mean_stacked_info, complete_D_train, empirical_covariances, \
                       len_train_clusters, n, pool, train_clusters_arr):
        optRes = [None for i in range(self.K)]
        for cluster in range(self.K):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                size_blocks = n
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, self.t_w * n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_info[self.K, cluster] = np.mean(D_train, axis=0)[(self.t_w - 1) * n:self.t_w * n].reshape([1, n])
                cluster_mean_stacked_info[self.K, cluster] = np.mean(D_train, axis=0)
                ##Fit a model - OPTIMIZATION
                probSize = self.t_w * size_blocks
                lamb = np.zeros((probSize, probSize)) + self._lambda
                S = np.cov(np.transpose(D_train))
                empirical_covariances[cluster] = S

                rho = 1
                solver = ADMMSolver(lamb, self.t_w, size_blocks, 1, S)
                # apply to process pool
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
        return optRes

    def stack_training_data(self, Data, n, num_train_points, training_indices):
        complete_D_train = np.zeros([num_train_points, self.t_w * n])
        for i in range(num_train_points):
            for k in range(self.t_w):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_D_train[i][k * n:(k + 1) * n] = Data[idx_k][0:n]
        return complete_D_train

    def prepare_out_directory(self):
        if self.mode == 0:
            output_path_for_one_para_comb = os.path.join(self.output_path, "lambda=" + str(self._lambda) + \
                                    " K=" + str(self.K) + " beta=" + str(self.beta) + " alpha=" + str(self.alpha) + \
                                    " t_w=" + str(self.t_w) + " r_w=" + str(self.r_w), \
                                    "profile", "r=" + str(self.r) + " (run " + str(self.r - self.r_w + 1) + " to run " + str(self.r) + ")")
        elif self.mode == 1:
            output_path_for_one_para_comb = os.path.join(self.output_path, "lambda=" + str(self._lambda) + \
                                    " K=" + str(self.K) + " beta=" + str(self.beta) + " alpha=" + str(self.alpha) + \
                                    " t_w=" + str(self.t_w) + " r_w=" + str(self.r_w), \
                                    "RCA_CTC", "r=" + str(self.r) + " (run " + str(self.r) + ")", "anomalous run data MRF")
        elif self.mode == 2:
            output_path_for_one_para_comb = os.path.join(self.output_path, "lambda=" + str(self._lambda) + \
                                    " K=" + str(self.K) + " beta=" + str(self.beta) + " alpha=" + str(self.alpha) + \
                                    " t_w=" + str(self.t_w) + " r_w=" + str(self.r_w), \
                                    "RCA_CTC", "r=" + str(self.r) + " (run " + str(self.r) + ")", "ground truth run data MRF", \
                                    "r=" + str(self.ground_truth_r) + " (run " + str(self.ground_truth_r) + ")")
        
        ### delete the folder and the content first
        if self.mode == 0:
            shutil.rmtree(path = output_path_for_one_para_comb, ignore_errors = True)
        
        if not os.path.exists(output_path_for_one_para_comb):
            try:
                os.makedirs(output_path_for_one_para_comb)
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise

        return output_path_for_one_para_comb

    def load_data(self, data):
        Data = np.loadtxt(data, delimiter=",")
        (m, n) = Data.shape  # m: num of observations, n: size of observation vector
        print("completed getting the data")
        return Data, m, n

    def log_parameters(self):
        if self.mode == 0:
            print("lambda = ", self._lambda)
            print("beta = ", self.beta)
            print("alpha = ", self.alpha)
            print("K = ", self.K)
            print("r_w = ", self.r_w)
            print("t_w = ", self.t_w)
            print("r = ", self.r)
            print("T = ", self.T)
        '''
        if self.mode == 1 or self.mode == 2:
            print("t = ", self.t)
        '''
    def predict_clusters(self, test_data = None):
        '''
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            np array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a np array!")
        else:
            test_data = self.trained_model['complete_D_train']

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['cluster_mean_info'],
                                                         self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         test_data,
                                                         self.trained_model['N'])

        # Update cluster points - using NEW smoothening
        #clustered_points = self.updateClusters(lle_all_points_clusters, beta=self.beta)
        clustered_points = self.updateClusters_with_GTC(lle_all_points_clusters)
        
        return(clustered_points)
    
    
    
    def getTrainTestSplit(self, m, num_blocks, num_stacked):
        '''
        - m: number of observations
        - num_blocks: t_w + 1
        - num_stacked: t_w
        Returns:
        - sorted list of training indices
        '''
        # Now splitting up stuff
        # split1 : Training and Test
        # split2 : Training and Test - different clusters
        training_percent = 1
        # list of training indices
        training_idx = np.random.choice(    \
            m-num_blocks+1, size=int((m-num_stacked)*training_percent), replace=False)
        # Ensure that the first and the last few points are in
        training_idx = list(training_idx)
        if 0 not in training_idx:
            training_idx.append(0)
        if m - num_stacked not in training_idx:
            training_idx.append(m-num_stacked)
        training_idx = np.array(training_idx)
        return sorted(training_idx)


    def upperToFull(self, a, eps=0):
            ind = (a < eps) & (a > -eps)
            a[ind] = 0
            n = int((-1 + np.sqrt(1 + 8*a.shape[0]))/2)
            A = np.zeros([n, n])
            A[np.triu_indices(n)] = a
            temp = A.diagonal()
            A = np.asarray((A + A.T) - np.diag(temp))
            return A


    def hex_to_rgb(self, value):
        """Return (red, green, blue) for the color given as #rrggbb."""
        lv = len(value)
        out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        out = tuple([x/256.0 for x in out])
        return out


    def updateClusters(self, LLE_node_vals, beta=1):
        """
        Takes in LLE_node_vals matrix and computes the path that minimizes
        the total cost over the path
        Note the LLE's are negative of the true LLE's actually!!!!!

        Note: switch penalty > 0
        """
        (T, num_clusters) = LLE_node_vals.shape
        future_cost_vals = np.zeros(LLE_node_vals.shape)

        # compute future costs
        for i in range(T-2, -1, -1):
            j = i+1
            indicator = np.zeros(num_clusters)
            future_costs = future_cost_vals[j, :]
            lle_vals = LLE_node_vals[j, :]
            for cluster in range(num_clusters):
                total_vals = future_costs + lle_vals + beta
                total_vals[cluster] -= beta
                future_cost_vals[i, cluster] = np.min(total_vals)

        # compute the best path
        path = np.zeros(T)

        # the first location
        curr_location = np.argmin(future_cost_vals[0, :] + LLE_node_vals[0, :])
        path[0] = curr_location

        # compute the path
        for i in range(T-1):
            j = i+1
            future_costs = future_cost_vals[j, :]
            lle_vals = LLE_node_vals[j, :]
            total_vals = future_costs + lle_vals + beta
            total_vals[int(path[i])] -= beta

            path[i+1] = np.argmin(total_vals)

        # return the computed path
        return path
    
    
    def updateClusters_with_GTC(self, ll_concatenate):
        #print(len(ll_concatenate))
        #print("T: " + str(self.T))
        def AssignPointsToClusters(ll):
        
            AllComb = [list(comb) for comb in itertools.product([k for k in range(self.K)], repeat = self.r_w)]
            #print(AllComb)

            PrevCost = [0] * len(AllComb)
            CurrCost = [float('Inf')] * len(AllComb)
            PrevPath = [ [] for _ in range(len(AllComb)) ]
            CurrPath = [ [] for _ in range(len(AllComb)) ]
            #print("PrevCost: " + str(PrevCost))
            #print("CurrCost: " + str(CurrCost))
            #print("PrevPath: " + str(PrevPath))
            #print("CurrPath: " + str(CurrPath))

            for t in range(self.T - self.t_w + 1):
                
                # print percentage
                if self.mode == 0:
                    if t%int((self.T - self.t_w + 1)/20) == 0:
                        percentage = int(t/int((self.T - self.t_w + 1)/20))
                        sys.stdout.write('\r')
                        # the exact output you're looking for:
                        sys.stdout.write("[%-20s] %d%%" % ('='*percentage, 5*percentage))
                        sys.stdout.flush()
                
                CurrCost = [float('Inf')] * len(AllComb)
                for CurrIdx in range(len(AllComb)):
                    for PrevIdx in range(len(AllComb)):
                        CurrCostPlusPrevCost = CalCost(PrevCost[PrevIdx], PrevIdx, CurrIdx, t, ll[t], AllComb)
                        if CurrCostPlusPrevCost < CurrCost[CurrIdx]:
                            CurrCost[CurrIdx] = CurrCostPlusPrevCost
                            CurrPath[CurrIdx] = deepcopy(PrevPath[PrevIdx])
                            CurrPath[CurrIdx].append(AllComb[CurrIdx])
                PrevCost = deepcopy(CurrCost)
                PrevPath = deepcopy(CurrPath)
            if self.mode == 0:
                print("\n")
            '''
            for Idx in range(len(AllComb)):
                print("CurrPath: " + str(CurrPath[Idx]) + " (CurrCost: " + str(CurrCost[Idx]) + ")")
            '''
            FinalMinIdx = CurrCost.index(min(CurrCost))
            FinalPath = CurrPath[FinalMinIdx]
            FinalCost = CurrCost[FinalMinIdx]
            #print("FinalPath: " + str(FinalPath))
            #print("FinalCost: " + str(CurrCost[FinalMinIdx]))

            return FinalPath, FinalCost
        
        def CalCost(PrevCost, PrevIdx, CurrIdx, t, ll, AllComb):
        
            CurrCost = PrevCost

            for r in range(self.r_w):
                CurrCost += ll[r][ AllComb[CurrIdx][r] ]
                #print(ll[r][ AllComb[CurrIdx][r] ])

            ### consider LTC
            for r in range(self.r_w):
                if t != 0 and AllComb[PrevIdx][r] != AllComb[CurrIdx][r]:
                    CurrCost += self.beta
                    #print("PLUS beta")

            ### consider GTC
            for r in range(self.r_w):
                if r != 0 and AllComb[CurrIdx][r-1] != AllComb[CurrIdx][r]:
                    CurrCost += self.alpha
                    #print("PLUS alpha")

            return CurrCost
        
        
        
        ll_concatenate = ll_concatenate.tolist()
        #print("len(ll_concatenate): " + str(len(ll_concatenate)))
        # change ll_concatenate to ll
        ll = [ [] for _ in range(self.T - self.t_w + 1) ]
        
        for t in range(self.T - self.t_w + 1):
            for ll_concatenate_idx in range((self.T - self.t_w + 1) * self.r_w):
                if ll_concatenate_idx % (self.T - self.t_w + 1) == t:
                    #print("t: " + str(t) + ", ll_concatenate_idx: " + str(ll_concatenate_idx))
                    ll[t].append(ll_concatenate[ll_concatenate_idx])
        #print("len(ll): " + str(len(ll)))
        
        ### call AssignPointsToClusters
        FinalPath, FinalCost = AssignPointsToClusters(ll)
        
        
        ### change FinalPath to FinalPath_concatenate
        FinalPath_concatenate = []

        for r in range(self.r_w):
            for t in range(self.T - self.t_w + 1):
                FinalPath_concatenate.append(FinalPath[t][r])
        
        return np.asarray(FinalPath_concatenate)
    
    
    def find_matching(self, confusion_matrix):
        """
        returns the perfect matching
        """
        _, n = confusion_matrix.shape
        path = []
        for i in range(n):
            max_val = -1e10
            max_ind = -1
            for j in range(n):
                if j in path:
                    pass
                else:
                    temp = confusion_matrix[i, j]
                    if temp > max_val:
                        max_val = temp
                        max_ind = j
            path.append(max_ind)
        return path


    def computeF1Score_delete(self, num_cluster, matching_algo, actual_clusters, threshold_algo, save_matrix=False):
        """
        computes the F1 scores and returns a list of values
        """
        F1_score = np.zeros(num_cluster)
        for cluster in range(num_cluster):
            matched_cluster = matching_algo[cluster]
            true_matrix = actual_clusters[cluster]
            estimated_matrix = threshold_algo[matched_cluster]
            if save_matrix: np.savetxt("estimated_matrix_cluster=" + str(
                cluster)+".csv", estimated_matrix, delimiter=",", fmt="%1.4f")
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for i in range(num_stacked*n):
                for j in range(num_stacked*n):
                    if estimated_matrix[i, j] == 1 and true_matrix[i, j] != 0:
                        TP += 1.0
                    elif estimated_matrix[i, j] == 0 and true_matrix[i, j] == 0:
                        TN += 1.0
                    elif estimated_matrix[i, j] == 1 and true_matrix[i, j] == 0:
                        FP += 1.0
                    else:
                        FN += 1.0
            precision = (TP)/(TP + FP)
            print("cluster #", cluster)
            print("TP,TN,FP,FN---------->", (TP, TN, FP, FN))
            recall = TP/(TP + FN)
            f1 = (2*precision*recall)/(precision + recall)
            F1_score[cluster] = f1
        return F1_score


    def compute_confusion_matrix(self, num_clusters, clustered_points_algo, sorted_indices_algo):
        """
        computes a confusion matrix and returns it
        """
        seg_len = 400
        true_confusion_matrix = np.zeros([num_clusters, num_clusters])
        for point in range(len(clustered_points_algo)):
            cluster = clustered_points_algo[point]
            num = (int(sorted_indices_algo[point]/seg_len) % num_clusters)
            true_confusion_matrix[int(num), int(cluster)] += 1
        return true_confusion_matrix


    def computeF1_macro(self, confusion_matrix, matching, num_clusters):
        """
        computes the macro F1 score
        confusion matrix : requres permutation
        matching according to which matrix must be permuted
        """
        # Permute the matrix columns
        permuted_confusion_matrix = np.zeros([num_clusters, num_clusters])
        for cluster in range(num_clusters):
            matched_cluster = matching[cluster]
            permuted_confusion_matrix[:, cluster] = confusion_matrix[:, matched_cluster]
    # Compute the F1 score for every cluster
        F1_score = 0
        for cluster in range(num_clusters):
            TP = permuted_confusion_matrix[cluster,cluster]
            FP = np.sum(permuted_confusion_matrix[:,cluster]) - TP
            FN = np.sum(permuted_confusion_matrix[cluster,:]) - TP
            precision = TP/(TP + FP)
            recall = TP/(TP + FN)
            f1 = stats.hmean([precision,recall])
            F1_score += f1
        F1_score /= num_clusters
        return F1_score

    def computeBIC(self, K, T, clustered_points, inverse_covariances, empirical_covariances):
        '''
        empirical covariance and inverse_covariance should be dicts
        K is num clusters
        T is num samples
        '''
        mod_lle = 0
        
        TICC_GTC_convergence_threshold = 2e-5
        clusterParams = {}
        for cluster, clusterInverse in inverse_covariances.items():
            mod_lle += np.log(np.linalg.det(clusterInverse)) - np.trace(np.dot(empirical_covariances[cluster], clusterInverse))
            clusterParams[cluster] = np.sum(np.abs(clusterInverse) > TICC_GTC_convergence_threshold)
        curr_val = -1
        non_zero_params = 0
        for val in clustered_points:
            if val != curr_val:
                non_zero_params += clusterParams[val]
                curr_val = val
        return non_zero_params * np.log(T) - 2*mod_lle

        
        
        
        
if __name__ == "__main__":
    
    ### get parameters form parameters.ini
    
    config = configparser.ConfigParser()
    config.read("parameters.ini")
    
    
    ### parameters for load data
    
    data_folder_path = eval(config.get("load_data", "data_folder_path"))
    tool_id = config.get("load_data", "tool_id")
    normalize = config.getboolean("load_data", "normalize")
    sample = config.getint("load_data", "sample")
    
    
    ### call load_data
    
    load_data_module = importlib.import_module("3-load_data")
    load_data_class = getattr(load_data_module, "load_data")
    load_data_instance = load_data_class(data_folder_path = data_folder_path, \
                                         tool_id = tool_id, \
                                         normalize = normalize, \
                                         sample = sample)
    data = load_data_instance()
    
    
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
    
    
    ### parameters for TICC_GTC_test
    
    mode = config.getint("TICC_GTC_test", "mode")
    r = eval(config.get("TICC_GTC_test", "r"))
    t = eval(config.get("TICC_GTC_test", "t"))
    _T = eval(config.get("TICC_GTC_test", "_T"))
    ground_truth_r = eval(config.get("TICC_GTC_test", "ground_truth_r"))
    
    
    ### call TICC_GTC
    
    TICC_GTC_instance = TICC_GTC(r_w = r_w, \
                                 t_w = t_w, \
                                 K = K, \
                                 _lambda = _lambda, \
                                 beta = beta, \
                                 alpha = alpha, \
                                 maxIters = maxIters, \
                                 TICC_GTC_convergence_threshold = TICC_GTC_convergence_threshold,\
                                 num_proc = num_proc, \
                                 output_path = output_path, \
                                 r = r, \
                                 t = t, \
                                 mode = mode, \
                                 T = _T, \
                                 ground_truth_r = ground_truth_r)

    if mode == 0:
        profile = TICC_GTC_instance.fit(data = data["time_series"][(r - r_w + 1) : (r) + 1])
    elif mode == 1:
        # execute twice
        anomalous_run_data_MRF = TICC_GTC_instance.fit(data = data["time_series"][r][(t - t_w + 1) : (t) + 1])
        
        t += 1
        TICC_GTC_instance = TICC_GTC(mode = mode, \
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
                                     r = r, \
                                     t = t, \
                                     T = _T, \
                                     ground_truth_r = ground_truth_r)
        anomalous_run_data_MRF = TICC_GTC_instance.fit(data = data["time_series"][r][(t - t_w + 1) : (t) + 1])
    elif mode == 2:
        # execute twice
        ground_truth_run_data_MRF = TICC_GTC_instance.fit(data = data["time_series"][ground_truth_r][(t - t_w + 1) : (t) + 1])
        
        ground_truth_r += 1
        TICC_GTC_instance = TICC_GTC(mode = mode, \
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
                                     r = r, \
                                     t = t, \
                                     T = _T, \
                                     ground_truth_r = ground_truth_r)
        ground_truth_run_data_MRF = TICC_GTC_instance.fit(data = data["time_series"][ground_truth_r][(t - t_w + 1) : (t) + 1])