import numpy as np
from scipy.stats import norm
from scipy.signal import find_peaks
from scipy.signal import convolve
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from scipy.stats import skew
from scipy import signal
from unidip import UniDip
import pandas as pd
from scipy.stats import mode
from scipy.stats import skew
from itertools import chain
import neo
import seaborn as sns
import quantities as pq
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.spike_train_synchrony import spike_contrast
from elephant.spike_train_correlation import spike_time_tiling_coefficient
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
from statistics import NormalDist
from scipy.spatial import distance

def k_means(burst_peaks, burst_borders, sdf, fs=12500):
    ibi = np.diff(burst_peaks)
    fr_at_peak = [sdf[(peak*fs).astype(int)] for peak in burst_peaks]
    duration = [burst[1]-burst[0] for burst in burst_borders]

    diff_length = len(fr_at_peak)+(len(ibi) - len(fr_at_peak))
    if diff_length != len(duration):
        duration = duration[1:]
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ibi, fr_at_peak[1:], duration)
    ax.set_xlabel("IBI")
    ax.set_ylabel("FR at peak")
    ax.set_zlabel("Duration")

    plt.figure()
    plt.scatter(ibi,fr_at_peak[1:])
    plt.xlabel("IBI")
    plt.ylabel("FR at peak")

    plt.figure()
    plt.plot(sdf)
    plt.plot((burst_peaks*fs).astype(int), sdf[(burst_peaks*fs).astype(int)], '.')
    plt.plot((prime_burst_peaks*fs).astype(int), sdf[(prime_burst_peaks*fs).astype(int)], 'x')

    for i in range(len(burst_borders)):
        plt.axvline((burst_borders[i][0] * fs).astype(int), color='cyan')
        plt.axvline((burst_borders[i][1] * fs).astype(int), color='blue')
    '''
    features = np.transpose([ibi, fr_at_peak[1:], duration])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 100,
        "random_state": 0
    }

    sse = []
    clusters = np.arange(1, 10, 1)
    for c in clusters:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=c, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
    ''' 
    # Identify ideal number of clusters
    plt.figure()
    plt.plot(range(1,10),sse)
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")

    kl = KneeLocator(clusters, sse, curve = "convex", direction = "decreasing")
    print(kl.elbow)
    '''
    kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
    kmeans.fit(scaled_features)

    x = np.transpose(scaled_features)[0]
    y = np.transpose(scaled_features)[1]
    z = np.transpose(scaled_features)[2]

    cluster_center_x = np.transpose(kmeans.cluster_centers_)[0, :]
    cluster_center_y = np.transpose(kmeans.cluster_centers_)[1, :]
    cluster_center_z = np.transpose(kmeans.cluster_centers_)[2, :]

    cluster_x_1 = x[kmeans.labels_==0]
    cluster_x_2 = x[kmeans.labels_==1]
    cluster_y_1 = y[kmeans.labels_==0]
    cluster_y_2 = y[kmeans.labels_==1]

    # Overlapping x
    rng_x = min(cluster_x_1.min(), cluster_x_2.min()), max(cluster_x_1.max(), cluster_x_2.max())
    n1_x, bins1_x = np.histogram(cluster_x_1, bins=10, range=rng_x)
    n2_x, bins2_x = np.histogram(cluster_x_2, bins=10, range=rng_x)
    intersection_x = np.minimum(n1_x,n2_x)

    overlapping_bins_x = np.nonzero(intersection_x)
    if len(overlapping_bins_x) > 0:
        overlapping_bursts_x = np.sum(n1_x[overlapping_bins_x] + n2_x[overlapping_bins_x])
        percent_overlapping_x = overlapping_bursts_x/len(x)
    else:
        percent_overlapping_x = 0

    # Overlapping y
    rng_y = min(cluster_y_1.min(), cluster_y_2.min()), max(cluster_y_1.max(), cluster_y_2.max())
    n1_y, bins1_y = np.histogram(cluster_y_1, bins=10, range=rng_y)
    n2_y, bins2_y = np.histogram(cluster_y_2, bins=10, range=rng_y)
    intersection_y = np.minimum(n1_y,n2_y)

    overlapping_bins_y = np.nonzero(intersection_y)
    if len(overlapping_bins_y) > 0:
        overlapping_bursts_y = np.sum(n1_y[overlapping_bins_y] + n2_y[overlapping_bins_y])
        percent_overlapping_y = overlapping_bursts_y/len(y)
    else:
        percent_overlapping_y = 0

    dst = distance.euclidean(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])

    if (percent_overlapping_x > 0.25) | (percent_overlapping_y > 0.25):
        print(f"Well is not reverberating.")
        return False, percent_overlapping_x, percent_overlapping_y, x, y, kmeans.labels_, kmeans.cluster_centers_, dst
    else:
        print(f"Well is reverberating.")
        return True, percent_overlapping_x, percent_overlapping_y, x, y, kmeans.labels_, kmeans.cluster_centers_, dst

    '''
    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', linewidth=2)

    palette = sns.color_palette("rocket")
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=x, y=y, palette = 'rocket', hue=kmeans.labels_)
    plt.xlabel("IBI (s, Standardized)")
    plt.ylabel("FR at peak (Hz, Standardized)")
    plt.legend([], [], frameon=False)
    #plt.savefig("Figures/Clustering/Reverberating_2DScatter_4.pdf", bbox_inches='tight', dpi=300,transparent=True)

    palette = sns.color_palette("rocket")
    color = [palette[(l+1)*2] for l in kmeans.labels_]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, color=color)
    ax.scatter(cluster_center_x, cluster_center_y, cluster_center_z, marker='X')
    ax.set_xlabel("IBI (s; Standardized)")
    ax.set_ylabel("FR at peak (Hz; Standardized)")
    ax.set_zlabel("Duration (s; Standardized)")
    #plt.savefig("Figures/Clustering/Reverberating_3DScatter_4.pdf", bbox_inches='tight', dpi=300,transparent=True)

    plt.figure(figsize=(6, 5))
    plt.eventplot(raster, color='black', linelengths=0.5, linewidths=0.75, alpha=0.35)
    plt.plot(np.arange(0,300,1/fs), sdf,'black')
    plt.xlim([40,120])
    plt.ylabel("Channel/Firing Rate (Hz; Standardized)")
    plt.xlabel("Time (s; Standardized)")
    #plt.savefig("Figures/Clustering/Reverberating_Raster_4.pdf", bbox_inches='tight', dpi=300,transparent=True)
    '''