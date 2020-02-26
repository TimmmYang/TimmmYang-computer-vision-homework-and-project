##
# Date: 2/25/2020
# Author: Lei Yang
# Description: Homework week 5 -- K-mean++
##

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs

class K_MEANS_CLUSTERING(object):
    def data_gen(self):
        center = [[10,10], [-10,-10], [10,-10], [0,0], [-10,10]]    # centers
        cluster_std = 2    # standard deviation
        X, labels = make_blobs(n_samples=100, centers=center, n_features=2, cluster_std=cluster_std, random_state=0)
        return X

    def distance(self, center, pts):
        d = np.square(pts[:, 0] - center[0]) + np.square(pts[:, 1] - center[1])
        return d

    def distance_2_points(self, p1, p2):   # should use np.sqrt (DO NOT USE SQUARE SUM DIRECTLY).  
        d = np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))
        return d
    
    def farthest_distance(self, pts, centers):
        d_max = 0  
        for pt in pts:
            d = 0
            for i in range(centers.__len__()):
                d += self.distance_2_points(pt, centers[i])
            if d > d_max:
                d_max = d
                p = pt
        return p

    def center_point(self, pts):
        center = np.array([np.mean(pts[:, 0]), np.mean(pts[:, 1])])
        return center
 
    def K_means_plus(self, X, k_clusters):
        n_points, n_features = X.shape
        center0_idx = np.random.randint(n_points)   # the first initial center index
        center_init = np.array([X[center0_idx, :]])
        for _ in range(k_clusters-1):  # for the remaining centers
            p = self.farthest_distance(X, center_init)
            center_init = np.concatenate([center_init, np.array([p])])   # add p to center_init 
        
        # display initial centers
        plt.figure()
        plt.title('The inital centers')
        plt.scatter(center_init[:, 0], center_init[:, 1])
        plt.show()
        # colors for animation
        col = ['HotPink', 'Aqua', 'LightSalmon', 'Chartreuse', 'yellow']
        # Initial distance from initial centers 
        d = np.zeros((k_clusters, n_points))
        for i in range(k_clusters):
            d[i, :] = self.distance(center_init[i,:], X) 

        # Initialization
        label = np.zeros(n_points)
        center = np.zeros([k_clusters,2])        
        flag = False
        count = 0
        while not flag:
            # clustering
            for i in range(n_points):
                label[i] = np.argmin(d[:, i])  # choose the min distance and label it
            center_pre = center.copy()   # use .copy() to avoid changing with center
            plt.ion()
            for i in range(k_clusters):
                # idx = np.argwhere(label == i).flatten()  # let idx be a 1d vector
                idx = label==i   # idx is a sequence of True and False, which can be used as index in data X 
                center[i, :] = self.center_point(X[idx, :])  # update center
                # dynamic show
                plt.scatter(center[i, 0], center[i, 1], linewidth=10, color=col[i])
                plt.scatter(X[idx, 0], X[idx, 1], color=col[i])
                plt.xlim((-15, 15))
                plt.ylim((-15, 15))
                plt.pause(0.8)
                plt.title("Clustering")
                d[i, :] = self.distance(center[i, :], X) # update distance  
            plt.ioff()
            plt.show()
            print('center points: ', center)    
            flag = np.array_equal(center, center_pre)
            count += 1
            if count > 30:
                break
        
        # result display
        
        for i in range(k_clusters):
            idx = label==i  
            plt.scatter(center[i, 0], center[i, 1], linewidth=10, color=col[i])
            plt.scatter(X[idx, 0], X[idx, 1], color = col[i])   
        plt.title('Clustering result')
        plt.show()

k_means = K_MEANS_CLUSTERING()
data = k_means.data_gen()

plt.figure()
plt.title('Raw Data')
plt.scatter(data[:, 0], data[:, 1])
plt.show()

k_clusters = 5
k_means.K_means_plus(data, k_clusters)
