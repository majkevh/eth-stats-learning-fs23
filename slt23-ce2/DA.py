import sklearn as skl
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize
import itertools
import random 

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """
    #4cluster Tmin = 1, alpha = .95, +100, T0 = 1e-5, epsilon = 1e-6, tol = 0 (default)
    #wine Tmin = 300, alpha = .98, *1.001, epsilon = 1e-4, T0 = 1e-5, tol != 0
    def __init__(self, n_clusters=8, 
                T_min = 1, 
                random_state=42, 
                alpha = 0.95, 
                threshold = 1e-6,
                tol = False,
                T0 = 1e-5,
                first_temperatur = True,
                metric="euclidian"):
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.T_min = T_min #minimal temperatur checked at step 5
        self.T0 = T0
        self.first_temperatur = first_temperatur
        self.tol = tol
        self.cluster_centers = None
        self.cluster_probs = None
        self.xi = 0.1 #perturbation of cluster variance
        self.alpha = alpha #cooling parameter
        self.epsilon =  threshold #step 4 hyperparameter
        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()
        self.bifurcation_tree = Tree()


    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """
        
        random.seed(self.random_state)
        n_samples = samples.shape[0]
        n_features = samples.shape[1]
        p_x = 1.0 / n_samples

        #1) Set limits: K_max --> self.n_clusters, T_min --> self.T_min

        #2) Initialize T, K, y_1
        cov_x = (1/n_samples)*np.dot(samples,samples.T) 
        eig, vec = np.linalg.eig(cov_x)

        if self.first_temperatur:
            T = 2*np.abs(np.max(eig))+100
        else:
            T = 2*np.abs(np.max(eig))*1.001

        K = 1

        y_1 = np.mean(samples, axis = 0) 
        self.cluster_centers = np.zeros((K, n_features))
        self.cluster_probs = np.ones((n_samples, K)) 
        self.cluster_centers[0] = y_1
        self.py_vector = np.ones(K)

        #Biforcation initialization: root note
        self.bifurcation_tree.create_node(tag = 0, 
                                          identifier = "0_0") #use identifier of type parent_child
        

        cluster_data = {"cluster_0": {"center": y_1, 
                                      "dist_to_root": [0.0], 
                                      "temperature": [T], 
                                      "offset": 0.0,
                                      "flip_direction": False}}

        while T> self.T0:

            #Perturb cluster centers if they are too close (see pseudocode pg20 skript 19)
            self.perturbe_centers(K, n_features)

            #3) Update 
            while  True: 
                old_cluster_centers = self.cluster_centers.copy()

                #Expectation step
                D = self.get_distance(samples, self.cluster_centers)
                self.cluster_probs = self._calculate_cluster_probs(D, T)

                #Maximization step
                self.py_vector = np.sum(self.cluster_probs, axis=0) * p_x
                self.cluster_centers =np.einsum('nk,nd->kd', self.cluster_probs, samples) * p_x / self.py_vector[:, np.newaxis]
               
                #4) Convergence test
                if np.linalg.norm(old_cluster_centers - self.cluster_centers) < self.epsilon :
                    break


            if T <= self.T0:		
                break	

            #5) Temperature drop below T_min check		            
            if T < self.T_min:		
                T = self.T0
            
            #Phase diagram plot update lists
            distortion = self.get_distorsion(samples)
            self.distortions.append(distortion)
            self.n_eff_clusters.append(K)
            self.temperatures.append(T)
            
            #6) Cooling step 
            T *= self.alpha

            #7) Phase transition
            if K < self.n_clusters:
                
                #Compute critical temperature for all clusters
                T_critics = np.array([self.critical_temperature(j, samples) for j in range(K)])

                for j in range(K):

                    #Check if critical temperature is reached for cluster j
                    if T < T_critics[j]:

                        #Add new codevector
                        split_cluster_center = self.cluster_centers[j, :].copy()
                        new_cluster_centroid =  split_cluster_center + np.random.normal(0, self.xi, n_features)
                        self.cluster_centers = np.vstack((self.cluster_centers, new_cluster_centroid))

                        #Update new cluster probabilities
                        self.py_vector[j] /= 2
                        self.py_vector = np.append(self.py_vector, self.py_vector[j].copy())
                        K+= 1

                        #Biforcation tree update
                        cluster_data = self.biforcate(j, K-1, cluster_data, split_cluster_center, T)

                        if K == self.n_clusters:
                            break

                    else:
                        cluster_data = self.update_cluster_data(cluster_data, j, T)
                        
            else:
                for j in range(K):
                    cluster_data = self.update_cluster_data(cluster_data, j, T)
        
        self.cluster_data_ = cluster_data



    def update_cluster_data(self, cluster_data, j, T):
        dist = np.linalg.norm(cluster_data[f"cluster_{j}"]["center"] - self.cluster_centers[j])
        cluster_data[f"cluster_{j}"]["dist_to_root"].append(cluster_data[f"cluster_{j}"]["offset"]+(1-2*int(cluster_data[f"cluster_{j}"]["flip_direction"]))*dist)
        cluster_data[f"cluster_{j}"]["temperature"].append(T)
        return cluster_data
    
    def biforcate(self, parent, child, cluster_data, new_cluster_centroid, T):
        idx = next((node.identifier for node in self.bifurcation_tree.leaves() if node.tag == parent), None)
        child_idx = int(idx[-1])+1
        self.bifurcation_tree.create_node(tag = parent, 
                                          identifier = str(parent)+'_'+str(child_idx),
                                          parent=idx) 
        self.bifurcation_tree.create_node(tag = child, 
                                          identifier = str(child)+'_'+str(child_idx), 
                                          parent=idx) 
        
        new_cluster_data = {
        "center": new_cluster_centroid,
        "dist_to_root": [cluster_data[f"cluster_{parent}"]["dist_to_root"][-1]],
        "offset": cluster_data[f"cluster_{parent}"]["dist_to_root"][-1],
        "temperature": [T],
        "flip_direction": False 
        }
        cluster_data[f"cluster_{child}"] = new_cluster_data

        cluster_data[f"cluster_{parent}"]["dist_to_root"].append(cluster_data[f"cluster_{parent}"]["dist_to_root"][-1])
        cluster_data[f"cluster_{parent}"]["offset"] = cluster_data[f"cluster_{parent}"]["dist_to_root"][-1]
        cluster_data[f"cluster_{parent}"]["temperature"].append(T)
        cluster_data[f"cluster_{parent}"]["flip_direction"] = True
        cluster_data[f"cluster_{parent}"]["center"] = new_cluster_centroid 

        return cluster_data
    

    def get_distorsion(self, samples):
        n_samples = samples.shape[0]
        D_flat = (np.square(self.get_distance(samples, self.cluster_centers))).reshape(-1)
        cluster_probs_flat = self.cluster_probs.flatten()
        return np.sum(cluster_probs_flat * D_flat) / n_samples
    
    def perturbe_centers(self, K, n_features):
        combs = itertools.combinations(range(K), 2)

        for a, b in combs:
            diff = self.cluster_centers[a] - self.cluster_centers[b]
            norm_diff = np.linalg.norm(diff)
    
            if norm_diff < self.xi:
                noise = np.random.normal(0, self.xi, size=n_features)
                self.cluster_centers[b] += noise

    def critical_temperature(self, j, samples):
        n_samples = samples.shape[0]
        prob = self.cluster_probs[:, j] / (self.py_vector[j] * n_samples)
        diff = samples - self.cluster_centers[j]
        outer = np.einsum('ij,ik->ijk', diff, diff)
        cov =  np.sum(prob[:, np.newaxis, np.newaxis] * outer, axis=0)

        eig , vec = np.linalg.eig(cov)
        return 2*np.abs(np.max((eig)))        

    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """  
        D = np.square(dist_mat)
        if self.tol:
            mins = np.min(D, axis=0, keepdims=True)
        else:
            mins = 0
        probs = self.py_vector*np.exp((-D+mins)/temperature)
        return normalize(probs, axis=1, norm='l1')
    
    def get_distance(self, samples, clusters):
        """Calculate the SQUARED distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """
        D = samples[:, np.newaxis, :] - clusters
        return np.sqrt(np.sum(D ** 2, axis=2))

        

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat
    

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting"""
        check_is_fitted(self, ["bifurcation_tree"])
        
        cut_idx = -20

        plt.figure(figsize=(10,5))
        for j in range(self.n_clusters):
            beta = [1/t for t in self.cluster_data_[f"cluster_{j}"]["temperature"]]
            plt.plot(self.cluster_data_[f"cluster_{j}"]["dist_to_root"][:cut_idx], beta[:cut_idx],
                     alpha=1, c='C%d' % int(j),
                     label='Cluster %d' % int(j))
            
        plt.legend()
        plt.xlabel("Distance to parent")
        plt.ylabel(r'$1/T$')
        plt.title('Bifurcation Plot')
        plt.show()
        
        return None


    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()
